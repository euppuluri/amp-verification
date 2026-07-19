"""
Predicts P(sequence is an antimicrobial peptide).

Trained offline (see train/train_amp_classifier.py) on:
  - positives: CAMPR3 + CAMPR4 known AMPs, APD3 validated AMPs
  - negatives: non-AMP peptides / random UniProt fragments
"""

from dataclasses import dataclass
import pickle
from pathlib import Path

MODEL_PATH = Path(__file__).parent.parent / "models" / "amp_classifier.pkl"


@dataclass
class AmpActivityResult:
    probability: float
    is_amp: bool


def _load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


_model = None


def predict_amp_probability(sequence: str, features) -> AmpActivityResult:
    global _model
    if _model is None:
        _model = _load_model()

    feature_vector = _features_to_vector(features)
    probability = float(_model.predict_proba([feature_vector])[0][1])

    return AmpActivityResult(
        probability=probability,
        is_amp=probability >= 0.5,
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
