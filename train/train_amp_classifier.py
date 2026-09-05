"""
Trains the binary AMP-activity classifier on CAMPR3 + CAMPR4 (positives)
vs. a negative set. APD3 is deliberately excluded — see
data/amp_reference_loader.py and train/validate_external.py.

Same homology-partitioned validation philosophy as the potency model:
cluster before splitting so the reported accuracy isn't inflated by
near-duplicate leakage.
"""

import pickle
from pathlib import Path
from pyexpat import model

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

from data.amp_reference_loader import build_amp_classifier_dataset
from pipeline.physicochemical import compute_physicochemical_features
from train.homology_split import homology_partition_split

MODEL_OUT_PATH = Path(__file__).parent.parent / "models" / "amp_classifier.pkl"

FEATURE_COLS = [
    "length", "net_charge", "hydrophobicity", "hydrophobic_moment",
    "helicity_score", "aggregation_propensity", "isoelectric_point", "protease_stability",
]


def main():
    df = build_amp_classifier_dataset()

    # Skip sequences with residues our feature formulas don't cover (X, B,
    # Z, U, O, etc.) instead of letting one bad row crash the whole run —
    # real reference databases occasionally contain these.
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

    df = pd.DataFrame(kept_rows).reset_index(drop=True)
    feature_df = pd.DataFrame(feature_rows)
    for col in FEATURE_COLS:
        df[col] = feature_df[col]

    train, test = homology_partition_split(df)

    X_train, y_train = train[FEATURE_COLS], train["label"]
    X_test, y_test = test[FEATURE_COLS], test["label"]

    model = RandomForestClassifier(n_estimators=300, random_state=42)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv)
    print(f"CV accuracy: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")

    model.fit(X_train, y_train)
    print(f"Held-out test accuracy: {model.score(X_test, y_test):.3f}")

    MODEL_OUT_PATH.parent.mkdir(exist_ok=True)
    with open(MODEL_OUT_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved model to {MODEL_OUT_PATH}")


if __name__ == "__main__":
    main()