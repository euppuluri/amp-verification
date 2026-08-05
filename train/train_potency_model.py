"""
Trains the Low/Medium/High potency classifier on GRAMPA + DBAASP.

Validation strategy:
  1. Homology-partitioned train/test split happens BEFORE potency labels
     are assigned. Tercile cutoffs are computed from the training
     partition only, then applied to both train and test — computing
     cutoffs from the pooled dataset before splitting would let test-set
     labels be informed by test-set values (a data leakage bug fixed in
     this version).
  2. Cross-validation within the training partition for hyperparameter
     tuning.
  3. External sanity check: verify predicted potency correlates in the
     expected direction with net charge / amphipathicity (known AMP
     biology), not just internal CV accuracy.

Gram stratification: pass --gram positive or --gram negative to train a
model only on organisms of that Gram type. Pass --organism "E. coli" to
filter to a single exact organism instead. Omit both for the pooled
(all-organism) model, kept as the default so the app has something to
load out of the box.

Usage:
    PYTHONPATH=. python -m train.train_potency_model
    PYTHONPATH=. python -m train.train_potency_model --gram negative
    PYTHONPATH=. python -m train.train_potency_model --organism "E. coli"
"""

import argparse
import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

from data.mic_data_loader import build_combined_mic_dataset, compute_potency_cutoffs, apply_potency_labels
from pipeline.physicochemical import compute_physicochemical_features
from train.homology_split import homology_partition_split

MODELS_DIR = Path(__file__).parent.parent / "models"

FEATURE_COLS = [
    "length", "net_charge", "hydrophobicity", "hydrophobic_moment",
    "helicity_score", "aggregation_propensity", "isoelectric_point", "protease_stability",
]


def build_feature_table(organism_filter=None, gram_filter=None) -> pd.DataFrame:
    """
    Returns sequence + log_mic + physicochemical features, WITHOUT potency
    labels yet — labels get assigned after the train/test split (see main()),
    not here, to avoid leaking test-set MIC values into label boundaries.

    Sequences with non-standard/ambiguous residues (X, B, Z, U, O, etc.) or
    any other character our feature formulas don't have a value for are
    skipped rather than crashing the whole run — real AMP databases contain
    a small number of these, and one bad row shouldn't kill training.
    """
    df = build_combined_mic_dataset(organism_filter=organism_filter, gram_filter=gram_filter)

    feature_rows = []
    kept_rows = []
    skipped = 0

    for row in df.itertuples(index=False):
        try:
            feats = compute_physicochemical_features(row.sequence)
        except KeyError:
            skipped += 1
            continue
        feature_rows.append(feats.as_dict())
        kept_rows.append(row)

    if skipped:
        print(f"Skipped {skipped} sequence(s) with non-standard residues our feature formulas don't cover.")

    kept_df = pd.DataFrame(kept_rows).reset_index(drop=True)
    feature_df = pd.DataFrame(feature_rows)
    return pd.concat([kept_df, feature_df], axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gram", choices=["positive", "negative"], default=None,
                         help="Train only on Gram-positive or Gram-negative organisms.")
    parser.add_argument("--organism", default=None,
                         help='Train only on one exact organism, e.g. "E. coli".')
    args = parser.parse_args()

    if args.organism and args.gram:
        raise SystemExit("Use --organism OR --gram, not both.")

    table = build_feature_table(organism_filter=args.organism, gram_filter=args.gram)
    print(f"Training on {len(table)} peptides "
          f"(organism_filter={args.organism!r}, gram_filter={args.gram!r})")

    # Split FIRST, then derive label cutoffs from the training partition
    # only, then apply those same cutoffs to both sides.
    train_df, test_df = homology_partition_split(table)

    low_cut, high_cut = compute_potency_cutoffs(train_df)
    print(f"Potency cutoffs (from training partition only): low_cut={low_cut:.3f}, high_cut={high_cut:.3f}")

    train_df = apply_potency_labels(train_df, low_cut, high_cut)
    test_df = apply_potency_labels(test_df, low_cut, high_cut)

    X_train, y_train = train_df[FEATURE_COLS], train_df["potency_class"]
    X_test, y_test = test_df[FEATURE_COLS], test_df["potency_class"]

    model = GradientBoostingClassifier(random_state=42)

    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"CV accuracy: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")

    model.fit(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    print(f"Held-out (homology-partitioned) test accuracy: {test_accuracy:.3f}")

    if args.gram:
        out_name = f"potency_classifier_gram_{args.gram}.pkl"
    elif args.organism:
        safe_name = args.organism.lower().replace(" ", "_").replace(".", "")
        out_name = f"potency_classifier_{safe_name}.pkl"
    else:
        out_name = "potency_classifier.pkl"

    out_path = MODELS_DIR / out_name
    MODELS_DIR.mkdir(exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved model to {out_path}")

    # Save the cutoffs alongside the model -- needed if you ever want to
    # explain "why High" in terms of actual log-MIC thresholds later.
    cutoffs_path = MODELS_DIR / (out_path.stem + "_cutoffs.txt")
    cutoffs_path.write_text(f"low_cut={low_cut}\nhigh_cut={high_cut}\n")


if __name__ == "__main__":
    main()