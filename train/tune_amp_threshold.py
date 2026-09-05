"""
Tunes AMP_PROBABILITY_THRESHOLD by sweeping candidate thresholds against
the held-out test set's TRUE labels, reporting precision/recall/F1 at
each -- rather than leaving the 0.7 default as an unexamined guess.

Precision here means: of sequences the model calls "AMP," what fraction
really are? Recall means: of true AMPs in the test set, what fraction
did the model catch? These trade off against each other as the threshold
moves -- there's no single "correct" answer, it depends on what mistake
matters more for your use case:
  - Higher threshold = fewer false "this is an AMP" calls (higher
    precision), but you'll miss some real AMPs (lower recall).
  - Lower threshold = catch more real AMPs (higher recall), but more
    non-AMPs incorrectly pass through (lower precision).

For a viability screening tool feeding into a "NOT VIABLE" hard-reject
decision (see decision.py), false negatives (missing a real AMP) are
arguably worse than false positives (a non-AMP proceeding to the next
check, where potency/toxicity screening still needs to pass) -- but
that's a judgment call for you and your mentor to make explicitly,
not something this script decides for you.

Usage:
    PYTHONPATH=. python -m train.tune_amp_threshold
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

from data.amp_reference_loader import build_amp_classifier_dataset
from pipeline.physicochemical import compute_physicochemical_features
from train.homology_split import homology_partition_split

FEATURE_COLS = [
    "length", "net_charge", "hydrophobicity", "hydrophobic_moment",
    "helicity_score", "aggregation_propensity", "isoelectric_point", "protease_stability",
]

CANDIDATE_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def main():
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

    df = pd.DataFrame(kept_rows).reset_index(drop=True)
    feature_df = pd.DataFrame(feature_rows)
    for col in FEATURE_COLS:
        df[col] = feature_df[col]

    train, test = homology_partition_split(df)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(train[FEATURE_COLS], train["label"])

    y_true = test["label"]
    y_proba = model.predict_proba(test[FEATURE_COLS])[:, 1]

    print(f"Evaluating on {len(test)} held-out test examples "
          f"({sum(y_true == 1)} true AMPs, {sum(y_true == 0)} true non-AMPs)\n")

    print(f"{'Threshold':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 44)
    for threshold in CANDIDATE_THRESHOLDS:
        y_pred = (y_proba >= threshold).astype(int)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        marker = "  <- current default (0.5 in sklearn, but decision.py uses 0.7)" if threshold == 0.7 else ""
        print(f"{threshold:>10.1f} {precision:>10.3f} {recall:>10.3f} {f1:>10.3f}{marker}")

    print("\nPick the threshold whose precision/recall tradeoff matches your")
    print("priorities, then update AMP_PROBABILITY_THRESHOLD in pipeline/decision.py.")


if __name__ == "__main__":
    main()