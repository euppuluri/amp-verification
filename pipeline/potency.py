"""
Predicts a Low / Medium / High potency class rather than a raw MIC value,
per the conclusion that exact MIC regression isn't reliable given
strain/media/protocol noise across GRAMPA and DBAASP.

Supports multiple trained model VARIANTS (pooled / Gram-positive /
Gram-negative), since train_potency_model.py can produce any of these
(see --gram flag). The caller picks which one to score against.

Class boundaries are defined at training time by binning the (log-)MIC
distribution of the TRAINING partition into terciles. See
train/train_potency_model.py.
"""

from dataclasses import dataclass
from enum import Enum
import pickle
from pathlib import Path
from typing import List, Dict

import pandas as pd

MODELS_DIR = Path(__file__).parent.parent / "models"

FEATURE_COLS = [
    "length", "net_charge", "hydrophobicity", "hydrophobic_moment",
    "helicity_score", "aggregation_propensity", "isoelectric_point", "protease_stability",
]

# Maps a user-facing variant name -> model filename. "pooled" is always
# available (it's what train_potency_model.py produces with no flags);
# the Gram variants only exist once you've run
# `train_potency_model.py --gram positive` / `--gram negative`.
MODEL_FILENAMES = {
    "pooled": "potency_classifier.pkl",
    "gram_negative": "potency_classifier_gram_negative.pkl",
    "gram_positive": "potency_classifier_gram_positive.pkl",
}

VARIANT_LABELS = {
    "pooled": "Pooled (all organisms)",
    "gram_negative": "Gram-negative targets (e.g. E. coli, P. aeruginosa)",
    "gram_positive": "Gram-positive targets (e.g. S. aureus, B. subtilis)",
}


class PotencyClass(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


@dataclass
class PotencyResult:
    potency_class: PotencyClass
    class_probabilities: Dict[str, float]  # {"Low": .1, "Medium": .2, "High": .7}
    model_variant_used: str  # the variant that actually got used (may differ
                              # from what was requested, if it fell back to pooled)


class PotencyModelUnavailable(Exception):
    """Raised when a requested model variant's .pkl file doesn't exist yet."""
    pass


_model_cache: Dict[str, object] = {}  # variant -> loaded sklearn model


def available_variants() -> Dict[str, bool]:
    """Returns {variant_name: is_trained} for every known variant, so the
    UI can show which options are actually usable right now."""
    return {
        variant: (MODELS_DIR / filename).exists()
        for variant, filename in MODEL_FILENAMES.items()
    }


def _load_model(variant: str):
    if variant not in MODEL_FILENAMES:
        raise ValueError(f"Unknown potency model variant: {variant!r}. "
                          f"Valid options: {list(MODEL_FILENAMES)}")

    if variant in _model_cache:
        return _model_cache[variant]

    path = MODELS_DIR / MODEL_FILENAMES[variant]
    if not path.exists():
        raise PotencyModelUnavailable(
            f"No trained model for variant {variant!r} at {path}. "
            f"Train it with: PYTHONPATH=. python -m train.train_potency_model --gram "
            f"{variant.replace('gram_', '')}"
        )

    with open(path, "rb") as f:
        model = pickle.load(f)
    _model_cache[variant] = model
    return model


def predict_potency_class(sequence: str, features, variant: str = "pooled",
                           fallback_to_pooled: bool = True) -> PotencyResult:
    """
    variant: "pooled", "gram_negative", or "gram_positive".
    fallback_to_pooled: if the requested variant isn't trained yet, silently
        fall back to the pooled model rather than erroring — set False if
        you want a hard failure instead (e.g. for offline analysis where a
        wrong fallback would be misleading).
    """
    try:
        model = _load_model(variant)
        variant_used = variant
    except PotencyModelUnavailable:
        if not fallback_to_pooled or variant == "pooled":
            raise
        model = _load_model("pooled")
        variant_used = "pooled"

    feature_row = pd.DataFrame([_features_to_vector(features)], columns=FEATURE_COLS)
    proba = model.predict_proba(feature_row)[0]
    labels = list(model.classes_)  # e.g. ["High", "Low", "Medium"]

    class_probabilities = dict(zip(labels, proba.tolist()))
    predicted_label = labels[proba.argmax()]

    return PotencyResult(
        potency_class=PotencyClass(predicted_label),
        class_probabilities=class_probabilities,
        model_variant_used=variant_used,
    )


def _features_to_vector(features) -> List[float]:
    """Must stay in sync with train_potency_model.py feature ordering."""
    return [
        features.length,
        features.net_charge,
        features.hydrophobicity,
        features.hydrophobic_moment,
        features.helicity_score,
        features.aggregation_propensity,
        features.isoelectric_point,
        features.protease_stability,
    ]