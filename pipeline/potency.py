"""
Predicts a Low / Medium / High potency class rather than a raw MIC value.

Class boundaries are defined at training time by binning the (log-)MIC
distribution of the combined GRAMPA + DBAASP training set into terciles.
See train/train_potency_model.py.
"""

from dataclasses import dataclass
from enum import Enum
import pickle
from pathlib import Path

MODEL_PATH = Path(__file__).parent.parent / "models" / "potency_classifier.pkl"


class PotencyClass(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


@dataclass
class PotencyResult:
    potency_class: PotencyClass
    class_probabilities: dict[str, float]


_model = None


def _load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def predict_potency_class(sequence: str, features) -> PotencyResult:
    global _model
    if _model is None:
        _model = _load_model()

    feature_vector = _features_to_vector(features)
    proba = _model.predict_proba([feature_vector])[0]
    labels = list(_model.classes_)

    class_probabilities = dict(zip(labels, proba.tolist()))
    predicted_label = labels[proba.argmax()]

    return PotencyResult(
        potency_class=PotencyClass(predicted_label),
        class_probabilities=class_probabilities,
    )


def _features_to_vector(features) -> list[float]:
    return [
        features.length,
        features.net_charge,
        features.hydrophobicity,
        features.hydrophobic_moment,
        features.helicity_score,
        features.aggregation_propensity,
        features.isoelectric_point,
    ]