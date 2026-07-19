"""
Combines AMP probability, toxicity, and potency class into a single
clinical-viability verdict with human-readable reasons.

Thresholds below are starting points — tune against real validation data.
"""

from dataclasses import dataclass
from enum import Enum

from pipeline.amp_activity import AmpActivityResult
from pipeline.toxicity import ToxicityResult
from pipeline.potency import PotencyResult, PotencyClass

AMP_PROBABILITY_THRESHOLD = 0.7
TOXICITY_SCORE_THRESHOLD = 0.5


class Verdict(str, Enum):
    VIABLE = "Viable for further testing"
    BORDERLINE = "Borderline — needs closer review"
    NOT_VIABLE = "Not viable"


@dataclass
class VerdictResult:
    verdict: Verdict
    reasons: list[str]


def make_viability_verdict(
    amp_result: AmpActivityResult,
    toxicity_result: ToxicityResult,
    potency_result: PotencyResult,
) -> VerdictResult:
    reasons = []

    if toxicity_result.is_toxic or toxicity_result.toxicity_score > TOXICITY_SCORE_THRESHOLD:
        reasons.append(
            f"Flagged as likely toxic by ToxinPred 3.0 (score={toxicity_result.toxicity_score:.2f})."
        )
        return VerdictResult(Verdict.NOT_VIABLE, reasons)

    if amp_result.probability < AMP_PROBABILITY_THRESHOLD:
        reasons.append(
            f"Low confidence this is an active AMP (P={amp_result.probability:.2f})."
        )
        return VerdictResult(Verdict.NOT_VIABLE, reasons)

    if potency_result.potency_class == PotencyClass.HIGH:
        reasons.append("High predicted potency class relative to GRAMPA/DBAASP peptides.")
        reasons.append(f"AMP confidence: {amp_result.probability:.2f}.")
        reasons.append(f"Toxicity score within acceptable range: {toxicity_result.toxicity_score:.2f}.")
        return VerdictResult(Verdict.VIABLE, reasons)

    if potency_result.potency_class == PotencyClass.MEDIUM:
        reasons.append("Medium predicted potency — passes AMP and toxicity checks but not top-tier potency.")
        return VerdictResult(Verdict.BORDERLINE, reasons)

    reasons.append("Low predicted potency class — unlikely to be competitive as a clinical candidate.")
    return VerdictResult(Verdict.NOT_VIABLE, reasons)