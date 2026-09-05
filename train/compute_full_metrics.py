"""
Computes AUROC, AUPRC, F1, and MCC on the AMP classifier's held-out test
set -- the metric suite your mentor specifically asked for, beyond the
plain accuracy number reported so far.

Why these specific metrics matter (for writing this up):
- AUROC: how well the model ranks true AMPs above non-AMPs across all
  possible thresholds, not just the one you picked.
- AUPRC: more informative than AUROC when classes are imbalanced (yours
  are, ~3:1) -- focuses on performance on the positive (AMP) class
  specifically.
- MCC (Matthews Correlation Coefficient): a single balanced summary that
  accounts for all four confusion-matrix cells, considered more reliable
  than F1 for imbalanced data by a lot of the ML literature.

Usage:
    PYTHONPATH=. python -m train.compute_full_metrics
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, matthews_corrcoef

from data.amp_reference_loader import build_amp_classifier_dataset
from pipeline.physicochemical import compute_physicochemical_features
from train.homology_split import homology_partition_split

FEATURE_COLS = ["length", "net_charge", "hydrophobicity", "hydrophobic_moment",
    "helicity_score", "aggregation_propensity", "isoelectric_point", "protease_stability"]


def main():
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

    train, test = homology_partition_split(df)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(train[FEATURE_COLS], train["label"])

    y_true = test["label"]
    y_proba = model.predict_proba(test[FEATURE_COLS])[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    print(f"Evaluated on {len(test)} held-out examples\n")
    print(f"AUROC: {roc_auc_score(y_true, y_proba):.4f}")
    print(f"AUPRC: {average_precision_score(y_true, y_proba):.4f}")
    print(f"F1:    {f1_score(y_true, y_pred):.4f}")
    print(f"MCC:   {matthews_corrcoef(y_true, y_pred):.4f}")
    print()
    print("Report all four of these in your manuscript, not just accuracy --")
    print("AUROC/AUPRC in particular are what reviewers will expect to see")
    print("for a classifier evaluated on imbalanced data.")


if __name__ == "__main__":
    main()