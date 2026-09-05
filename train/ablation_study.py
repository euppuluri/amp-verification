"""
Ablation study: which of the 8 physicochemical features actually drive
the AMP classifier's predictions? Directly addresses your mentor's
request for "ablation experiments showing which components contribute
most."

Uses PERMUTATION importance (not the model's built-in feature_importances_)
because permutation importance is measured on the held-out TEST set and
reflects real predictive contribution, whereas built-in importances are
computed from training data and can be misleading (e.g. biased toward
high-cardinality features).

Method: for each feature, shuffle its values across the test set (breaking
its relationship with the label) and measure how much accuracy drops. A
big drop = the model relies on that feature heavily. Near-zero drop = the
model barely uses it.

Usage:
    PYTHONPATH=. python -m train.ablation_study
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

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

    result = permutation_importance(
        model, test[FEATURE_COLS], test["label"],
        n_repeats=30, random_state=42, scoring="accuracy",
    )

    ranked = sorted(zip(FEATURE_COLS, result.importances_mean, result.importances_std),
                     key=lambda x: x[1], reverse=True)

    print(f"{'Feature':25s} {'Importance':>12s} {'Std Dev':>10s}")
    print("-" * 49)
    for name, mean, std in ranked:
        print(f"{name:25s} {mean:12.4f} {std:10.4f}")

    print()
    print("Importance = average accuracy drop when this feature is shuffled.")
    print("Features near 0.0000 are contributing little -- worth discussing")
    print("in your manuscript as candidates to simplify or replace, and")
    print("features at the top are your strongest evidence for WHY the")
    print("classifier works, not just THAT it works.")


if __name__ == "__main__":
    main()