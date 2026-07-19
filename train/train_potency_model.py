"""
Trains the Low/Medium/High potency classifier on GRAMPA + DBAASP.
"""

import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

from data.mic_data_loader import build_combined_mic_dataset, assign_potency_labels
from pipeline.physicochemical import compute_physicochemical_features
from train.homology_split import homology_partition_split

MODEL_OUT_PATH = Path(__file__).parent.parent / "models" / "potency_classifier.pkl"


def build_training_table() -> pd.DataFrame:
    df = build_combined_mic_dataset()
    df = assign_potency_labels(df)

    feature_rows = []
    for seq in df["sequence"]:
        feats = compute_physicochemical_features(seq)
        feature_rows.append(feats.as_dict())

    feature_df = pd.DataFrame(feature_rows)
    return pd.concat([df.reset_index(drop=True), feature_df], axis=1)


def main():
    table = build_training_table()
    train_df, test_df = homology_partition_split(table)

    feature_cols = [
        "length", "net_charge", "hydrophobicity", "hydrophobic_moment",
        "helicity_score", "aggregation_propensity", "isoelectric_point",
    ]

    X_train, y_train = train_df[feature_cols], train_df["potency_class"]
    X_test, y_test = test_df[feature_cols], test_df["potency_class"]

    model = GradientBoostingClassifier(random_state=42)

    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"CV accuracy: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")

    model.fit(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    print(f"Held-out (homology-partitioned) test accuracy: {test_accuracy:.3f}")

    MODEL_OUT_PATH.parent.mkdir(exist_ok=True)
    with open(MODEL_OUT_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved model to {MODEL_OUT_PATH}")


if __name__ == "__main__":
    main()