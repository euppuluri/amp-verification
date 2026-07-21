"""
Candidacy Score: a single 0-1 number combining AMP probability, toxicity,
potency, and protease stability into one composite signal.

IMPORTANT — why this is a weighted formula, not a trained model:
A trained meta-model needs labeled ground truth (e.g. "did this peptide
succeed in clinical testing: yes/no"). None of GRAMPA, DBAASP, CAMPR, or
APD3 contain that label — they contain lab-measured MIC/toxicity/activity,
not clinical outcomes. Training a model implies it learned real patterns
from data; without an outcome label, there's nothing to learn from, and a
"trained" model here would just be an opaque way of hard-coding weights
anyway. This module hard-codes those same weights transparently instead,
so you can see and adjust exactly what's driving the score.

The weights below are a reasonable starting point, not a validated
formula — tune them if you have domain expertise on how these four
factors should trade off for your specific use case.
"""

from dataclasses import dataclass

from pipeline.amp_activity import AmpActivityResult
from pipeline.toxicity import ToxicityResult
from pipeline.potency import PotencyResult

# Weights must sum to 1.0
WEIGHT_AMP = 0.30
WEIGHT_TOXICITY = 0.30   # inverted: low toxicity contributes positively
WEIGHT_POTENCY = 0.30
WEIGHT_STABILITY = 0.10

POTENCY_SCORE_MAP = {
    "Low": 0.0,
    "Medium": 0.5,
    "High": 1.0,
}


@dataclass
class CandidacyScore:
    score: float  # 0-1, higher = more promising overall candidate
    breakdown: dict  # component contributions, for transparency


def compute_candidacy_score(
    amp_result: AmpActivityResult,
    toxicity_result: ToxicityResult,
    potency_result: PotencyResult,
    protease_stability: float,
) -> CandidacyScore:
    amp_component = amp_result.probability
    toxicity_component = 1.0 - min(toxicity_result.toxicity_score, 1.0)  # invert: lower toxicity = better

    # Use the potency model's own class probabilities as a continuous
    # signal, weighted by how "good" each class is, rather than just the
    # single predicted label — this uses the full distribution the model
    # already computed instead of throwing information away.
    potency_component = sum(
        POTENCY_SCORE_MAP[label] * prob
        for label, prob in potency_result.class_probabilities.items()
    )

    stability_component = protease_stability

    score = (
        WEIGHT_AMP * amp_component
        + WEIGHT_TOXICITY * toxicity_component
        + WEIGHT_POTENCY * potency_component
        + WEIGHT_STABILITY * stability_component
    )

    breakdown = {
        "amp_contribution": round(WEIGHT_AMP * amp_component, 3),
        "toxicity_contribution": round(WEIGHT_TOXICITY * toxicity_component, 3),
        "potency_contribution": round(WEIGHT_POTENCY * potency_component, 3),
        "stability_contribution": round(WEIGHT_STABILITY * stability_component, 3),
    }

    return CandidacyScore(score=round(score, 3), breakdown=breakdown)