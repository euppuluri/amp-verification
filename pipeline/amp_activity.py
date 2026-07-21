"""
Predicts P(sequence is an antimicrobial peptide).

Trained offline (see train/train_amp_classifier.py) on:
  - positives: CAMPR3 + CAMPR4 known AMPs, APD3 validated AMPs
  - negatives: non-AMP peptides / random UniProt fragments
"""

from dataclasses import dataclass
import pickle
from pathlib import Path
from typing import List

import pandas as pd

MODEL_PATH = Path(__file__).parent.parent / "models" / "amp_classifier.pkl"

FEATURE_COLS = [
    "length", "net_charge", "hydrophobicity", "hydrophobic_moment",
    "helicity_score", "aggregation_propensity", "isoelectric_point",
]


@dataclass
class AmpActivityResult:
    probability: float   # 0-1
    is_amp: bool          # thresholded at 0.5


def _load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


_model = None  # lazy-loaded singleton


def predict_amp_probability(sequence: str, features) -> AmpActivityResult:
    global _model
    if _model is None:
        _model = _load_model()

    feature_row = pd.DataFrame([_features_to_vector(features)], columns=FEATURE_COLS)
    probability = float(_model.predict_proba(feature_row)[0][1])

    return AmpActivityResult(
        probability=probability,
        is_amp=probability >= 0.5,
    )


def _features_to_vector(features) -> List[float]:
    """Flatten PhysicochemicalFeatures into the exact column order the
    trained model expects. Must stay in sync with train_amp_classifier.py."""
    return [
        features.length,
        features.net_charge,
        features.hydrophobicity,
        features.hydrophobic_moment,
        features.helicity_score,
        features.aggregation_propensity,
        features.isoelectric_point,
    ]