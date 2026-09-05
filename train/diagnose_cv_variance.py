"""
Diagnoses why the AMP classifier's cross-validation accuracy (71.7% +/-
18.4%) sits oddly below both held-out test accuracy (88.4%) and external
APD3 validation recall (95.0%). High variance across folds with a LOWER
mean than held-out/external results is unusual and worth understanding
before reporting these numbers anywhere.

This prints several diagnostics rather than assuming a single cause:
  1. Per-fold scores (not just mean/std) -- is variance from one bad
     outlier fold, or genuinely spread across all folds?
  2. Per-fold class balance -- did stratification actually work?
  3. Per-fold size -- are folds roughly equal?
  4. Comparison: default (unshuffled) CV vs explicitly shuffled CV --
     tests whether fold order matters for your specific data.
  5. A repeated-CV run (multiple different random splits) -- distinguishes
     "this specific split was unlucky" from "the model is genuinely
     unstable on this data."

Usage:
    PYTHONPATH=. python -m train.diagnose_cv_variance
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, cross_val_score

from data.amp_reference_loader import build_amp_classifier_dataset
from pipeline.physicochemical import compute_physicochemical_features
from train.homology_split import homology_partition_split

FEATURE_COLS = [
    "length", "net_charge", "hydrophobicity", "hydrophobic_moment",
    "helicity_score", "aggregation_propensity", "isoelectric_point", "protease_stability",
]


def build_training_data():
    df = build_amp_classifier_dataset()
    feature_rows = []
    kept_rows = []
    for row in df.itertuples(index=False):
        try:
            feats = compute_physicochemical_features(row.sequence)
        except KeyError:
            continue
        feature_rows.append(feats.as_dict())
        kept_rows.append(row)

    import pandas as pd
    df = pd.DataFrame(kept_rows).reset_index(drop=True)
    feature_df = pd.DataFrame(feature_rows)
    for col in FEATURE_COLS:
        df[col] = feature_df[col]

    train, _ = homology_partition_split(df)
    return train[FEATURE_COLS], train["label"]


def main():
    X, y = build_training_data()
    print(f"Training partition: {len(X)} examples, {sum(y == 1)} positive, {sum(y == 0)} negative\n")

    model = RandomForestClassifier(n_estimators=100, random_state=42)

    print("=" * 60)
    print("1. DEFAULT cross_val_score(cv=5) -- what train_amp_classifier.py runs")
    print("=" * 60)
    default_scores = cross_val_score(model, X, y, cv=5)
    print(f"Per-fold scores: {[round(s, 3) for s in default_scores]}")
    print(f"Mean: {default_scores.mean():.3f} +/- {default_scores.std():.3f}\n")

    print("=" * 60)
    print("2. EXPLICIT StratifiedKFold(shuffle=True) -- controls for fold order")
    print("=" * 60)
    cv_shuffled = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    shuffled_scores = cross_val_score(model, X, y, cv=cv_shuffled)
    print(f"Per-fold scores: {[round(s, 3) for s in shuffled_scores]}")
    print(f"Mean: {shuffled_scores.mean():.3f} +/- {shuffled_scores.std():.3f}\n")

    print("=" * 60)
    print("3. Per-fold class balance and size (using the shuffled split)")
    print("=" * 60)
    for i, (train_idx, val_idx) in enumerate(cv_shuffled.split(X, y)):
        val_y = y.iloc[val_idx]
        pos_frac = (val_y == 1).mean()
        print(f"  Fold {i}: {len(val_idx)} examples, {pos_frac:.1%} positive")

    print()
    print("=" * 60)
    print("4. REPEATED CV (5 folds x 3 repeats = 15 total scores)")
    print("   Distinguishes 'this split was unlucky' from 'genuinely unstable'")
    print("=" * 60)
    repeated_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
    repeated_scores = cross_val_score(model, X, y, cv=repeated_cv)
    print(f"All {len(repeated_scores)} scores: {[round(s, 3) for s in repeated_scores]}")
    print(f"Mean: {repeated_scores.mean():.3f} +/- {repeated_scores.std():.3f}")
    print(f"Min: {repeated_scores.min():.3f}, Max: {repeated_scores.max():.3f}")

    print()
    print("=" * 60)
    print("INTERPRETATION GUIDE")
    print("=" * 60)
    print("- If shuffled (step 2) has much lower variance than default (step 1):")
    print("  fold ORDER matters for your data -- the training DataFrame has")
    print("  structure (e.g. CAMPR3 rows before CAMPR4 rows) that the default")
    print("  unshuffled CV doesn't handle well. Fix: always pass an explicit")
    print("  shuffled StratifiedKFold to cross_val_score from now on.")
    print("- If step 3 shows very uneven class balance across folds despite")
    print("  'stratified' splitting: investigate further, this shouldn't happen.")
    print("- If repeated CV (step 4) still shows high variance across 15 runs:")
    print("  the instability is real, not just one unlucky split -- may indicate")
    print("  the model or features genuinely struggle on certain data subsets.")


if __name__ == "__main__":
    main()