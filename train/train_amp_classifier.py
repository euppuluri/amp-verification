"""
Trains the binary AMP-activity classifier on CAMPR3 + CAMPR4 + APD3
(positives) vs. a negative set.
"""

import pickle
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from data.amp_reference_loader import build_amp_classifier_dataset
from pipeline.physicochemical import compute_physicochemical_features
from train.homology_split import homology_partition_split

MODEL_OUT_PATH = Path(__file__).parent.parent / "models" / "amp_classifier.pkl"

FEATURE_COLS = [
    "length", "net_charge", "hydrophobicity", "hydrophobic_moment",
    "helicity_score", "aggregation_propensity", "isoelectric_point",
]


def main():
    df = build_amp_classifier_dataset()

    feature_rows = [compute_physicochemical_features(s).as_dict() for s in df["sequence"]]
    for col in FEATURE_COLS:
        df[col] = [row[col] for row in feature_rows]

    train, test = homology_partition_split(df)

    X_train, y_train = train[FEATURE_COLS], train["label"]
    X_test, y_test = test[FEATURE_COLS], test["label"]

    model = RandomForestClassifier(n_estimators=300, random_state=42)

    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"CV accuracy: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")

    model.fit(X_train, y_train)
    print(f"Held-out test accuracy: {model.score(X_test, y_test):.3f}")

    MODEL_OUT_PATH.parent.mkdir(exist_ok=True)
    with open(MODEL_OUT_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved model to {MODEL_OUT_PATH}")


if __name__ == "__main__":
    main()