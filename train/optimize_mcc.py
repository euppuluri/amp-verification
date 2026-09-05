"""
Searches for a training configuration (class_weight) and decision
threshold that improve MCC -- WITHOUT touching the final held-out test
set during the search, to avoid test-set contamination.

Methodology: splits the training partition further into a sub-train and
validation set. Tries a few class_weight configurations, and for each,
sweeps thresholds on the VALIDATION set only to find the MCC-optimal
cutoff. The single best (class_weight, threshold) combination found on
validation is then evaluated exactly ONCE on the real held-out test set
for the final, honestly-reported number.

Usage:
    PYTHONPATH=. python -m train.optimize_mcc
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef, roc_auc_score, f1_score

from data.amp_reference_loader import build_amp_classifier_dataset
from pipeline.physicochemical import compute_physicochemical_features
from train.homology_split import homology_partition_split

FEATURE_COLS = ["length", "net_charge", "hydrophobicity", "hydrophobic_moment",
    "helicity_score", "aggregation_propensity", "isoelectric_point", "protease_stability"]

CLASS_WEIGHT_OPTIONS = [None, "balanced", {0: 2, 1: 1}, {0: 3, 1: 1}]
THRESHOLD_CANDIDATES = np.arange(0.1, 0.95, 0.05)


def build_dataset():
    df = build_amp_classifier_dataset()
    rows = []
    kept = []
    for row in df.itertuples(index=False):
        try:
            feats = compute_physicochemical_features(row.sequence)
        except KeyError:
            continue
        rows.append(feats.as_dict())
        kept.append(row)
    df = pd.DataFrame(kept).reset_index(drop=True)
    fdf = pd.DataFrame(rows)
    for c in FEATURE_COLS:
        df[c] = fdf[c]
    return df


def main():
    df = build_dataset()

    # Outer split: train_full vs. the REAL held-out test set (touched only once, at the end)
    train_full, test = homology_partition_split(df, test_frac=0.2)
    # Inner split: sub-train vs. validation, used ONLY for tuning
    sub_train, val = homology_partition_split(train_full, test_frac=0.2)

    print(f"Sub-train: {len(sub_train)}, Validation: {len(val)}, Test (untouched until the end): {len(test)}\n")

    best_config = None
    best_val_mcc = -1

    print(f"{'class_weight':20s} {'threshold':>10s} {'val MCC':>10s}")
    print("-" * 42)

    for class_weight in CLASS_WEIGHT_OPTIONS:
        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight=class_weight)
        model.fit(sub_train[FEATURE_COLS], sub_train["label"])
        val_proba = model.predict_proba(val[FEATURE_COLS])[:, 1]

        best_threshold_for_this_weight = None
        best_mcc_for_this_weight = -1

        for threshold in THRESHOLD_CANDIDATES:
            val_pred = (val_proba >= threshold).astype(int)
            mcc = matthews_corrcoef(val["label"], val_pred)
            if mcc > best_mcc_for_this_weight:
                best_mcc_for_this_weight = mcc
                best_threshold_for_this_weight = threshold
            if mcc > best_val_mcc:
                best_val_mcc = mcc
                best_config = (class_weight, threshold)

        print(f"{str(class_weight):20s} {best_threshold_for_this_weight:>10.2f} {best_mcc_for_this_weight:>10.4f}")

    print(f"\nBest found on VALIDATION: class_weight={best_config[0]}, threshold={best_config[1]:.2f} "
          f"(val MCC={best_val_mcc:.4f})")

    # Final, honest evaluation: retrain on ALL training data (sub_train + val) with the
    # chosen class_weight, then evaluate ONCE on the untouched test set.
    final_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight=best_config[0])
    final_model.fit(train_full[FEATURE_COLS], train_full["label"])

    test_proba = final_model.predict_proba(test[FEATURE_COLS])[:, 1]
    test_pred = (test_proba >= best_config[1]).astype(int)

    print(f"\n{'=' * 50}")
    print("FINAL, HONEST TEST-SET RESULTS (test set untouched during tuning)")
    print(f"{'=' * 50}")
    print(f"MCC:   {matthews_corrcoef(test['label'], test_pred):.4f}")
    print(f"F1:    {f1_score(test['label'], test_pred):.4f}")
    print(f"AUROC: {roc_auc_score(test['label'], test_proba):.4f}")
    print(f"\nCompare this MCC against your original 0.7107 (class_weight=None, threshold=0.5)")
    print(f"to see the real improvement from this tuning.")


if __name__ == "__main__":
    main()