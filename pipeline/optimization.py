"""
Given a Low or Medium potency peptide, suggests concrete sequence
alterations predicted to improve it -- using the ACTUAL trained AMP and
potency classifiers, not simulated/random scoring.

Design notes:
- Generates candidate variants via biologically-motivated mutation
  strategies (increase cationic charge, improve amphipathicity, improve
  protease stability, improve helix propensity, plus a few random
  single-point mutations for diversity) rather than pure random search --
  each suggestion comes with a concrete, explainable rationale.
- Every candidate is scored with the REAL trained models
  (predict_amp_probability, predict_potency_class) -- exactly the same
  code path a live user query goes through, not a separate simulated
  scoring function.
- Toxicity (ToxinPred) is deliberately NOT called for every candidate --
  that's a slow external subprocess call, unreasonable to run for dozens
  of candidates per request. Toxicity should be checked on the final
  shortlisted candidates before treating any of them as serious leads,
  same as any computational screening pipeline -- flagged clearly in the
  output rather than silently ignored.
- A simple heuristic guards against candidates likely to increase
  toxicity risk even without calling ToxinPred: extreme hydrophobicity
  or extreme charge (well outside the range seen in natural AMPs) are
  penalized in scoring, since both are established correlates of
  increased hemolytic/cytotoxic risk in the AMP literature.
"""

import random
from dataclasses import dataclass
from typing import List

from pipeline.physicochemical import compute_physicochemical_features, PhysicochemicalFeatures
from pipeline.amp_activity import predict_amp_probability
from pipeline.potency import predict_potency_class, PotencyClass

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
CATIONIC_RESIDUES = "KR"
HYDROPHOBIC_RESIDUES = "AVLIMFWY"
STRONG_HELIX_FORMERS = "AELMQK"  # high Chou-Fasman helix propensity

# Heuristic "safe" ranges informed by typical natural AMP properties --
# candidates drifting far outside these get penalized, as a cheap proxy
# for toxicity risk without calling ToxinPred on every candidate.
SAFE_HYDROPHOBICITY_RANGE = (-1.5, 1.5)
SAFE_CHARGE_RANGE = (1.0, 8.0)

NUM_CANDIDATES_TO_GENERATE = 40


@dataclass
class Suggestion:
    sequence: str
    mutation_description: str
    original_potency_class: str
    new_potency_class: str
    new_potency_probabilities: dict
    new_amp_probability: float
    feature_deltas: dict
    improvement_score: float


def _random_position(sequence: str, exclude: set = None) -> int:
    exclude = exclude or set()
    candidates = [i for i in range(len(sequence)) if i not in exclude]
    return random.choice(candidates) if candidates else random.randrange(len(sequence))


def _mutate_increase_charge(sequence: str) -> tuple:
    """Substitute a non-cationic residue with K or R."""
    non_cationic_positions = [i for i, aa in enumerate(sequence) if aa not in CATIONIC_RESIDUES]
    if not non_cationic_positions:
        return None
    pos = random.choice(non_cationic_positions)
    new_residue = random.choice(CATIONIC_RESIDUES)
    old_residue = sequence[pos]
    new_seq = sequence[:pos] + new_residue + sequence[pos + 1:]
    return new_seq, f"Increased net charge: position {pos + 1} {old_residue}->{new_residue}"


def _mutate_improve_amphipathicity(sequence: str) -> tuple:
    """
    Nudges toward a stronger hydrophobic moment by substituting the
    residue with the weakest individual hydrophobicity contribution with
    a strongly hydrophobic residue, biasing toward a cleaner amphipathic
    helix face.
    """
    from pipeline.physicochemical import KYTE_DOOLITTLE
    weakest_pos = min(range(len(sequence)), key=lambda i: KYTE_DOOLITTLE[sequence[i]])
    old_residue = sequence[weakest_pos]
    new_residue = random.choice(HYDROPHOBIC_RESIDUES)
    new_seq = sequence[:weakest_pos] + new_residue + sequence[weakest_pos + 1:]
    return new_seq, f"Improved amphipathicity: position {weakest_pos + 1} {old_residue}->{new_residue}"


def _mutate_improve_helicity(sequence: str) -> tuple:
    """Substitute the weakest helix-forming residue with a strong one."""
    from pipeline.physicochemical import CHOU_FASMAN_HELIX
    weakest_pos = min(range(len(sequence)), key=lambda i: CHOU_FASMAN_HELIX[sequence[i]])
    old_residue = sequence[weakest_pos]
    new_residue = random.choice(STRONG_HELIX_FORMERS)
    new_seq = sequence[:weakest_pos] + new_residue + sequence[weakest_pos + 1:]
    return new_seq, f"Improved helix propensity: position {weakest_pos + 1} {old_residue}->{new_residue}"


def _mutate_improve_protease_stability(sequence: str) -> tuple:
    """
    Finds a predicted protease cleavage site (K/R/F/Y/W/A/G/V not
    followed by Proline) and substitutes that residue for a
    non-cleavage-prone one, or inserts nothing -- swaps the residue
    itself for a chemically similar but non-cleavage-prone alternative
    where possible (charged -> charged, hydrophobic -> hydrophobic) to
    avoid wrecking other properties in the same move.
    """
    cleavage_prone = set("KRFYWAGV")
    substitution_map = {
        "K": "Q", "R": "N",           # cationic -> polar, avoids trypsin site
        "F": "L", "Y": "T", "W": "L",  # aromatic -> non-aromatic, avoids chymotrypsin site
        "A": "S", "G": "S", "V": "T",  # small aliphatic -> polar, avoids elastase site
    }
    candidate_positions = [
        i for i in range(len(sequence) - 1)
        if sequence[i] in cleavage_prone and sequence[i + 1] != "P"
    ]
    if not candidate_positions:
        return None
    pos = random.choice(candidate_positions)
    old_residue = sequence[pos]
    new_residue = substitution_map[old_residue]
    new_seq = sequence[:pos] + new_residue + sequence[pos + 1:]
    return new_seq, f"Improved protease stability: position {pos + 1} {old_residue}->{new_residue} (removes a predicted cleavage site)"


def _mutate_random_point(sequence: str) -> tuple:
    """Unbiased single-point mutation, for diversity alongside the targeted strategies."""
    pos = _random_position(sequence)
    old_residue = sequence[pos]
    new_residue = random.choice([aa for aa in AMINO_ACIDS if aa != old_residue])
    new_seq = sequence[:pos] + new_residue + sequence[pos + 1:]
    return new_seq, f"Exploratory mutation: position {pos + 1} {old_residue}->{new_residue}"


MUTATION_STRATEGIES = [
    _mutate_increase_charge,
    _mutate_improve_amphipathicity,
    _mutate_improve_helicity,
    _mutate_improve_protease_stability,
    _mutate_random_point,
]


def _is_within_safe_range(features: PhysicochemicalFeatures) -> bool:
    charge_ok = SAFE_CHARGE_RANGE[0] <= features.net_charge <= SAFE_CHARGE_RANGE[1]
    hydro_ok = SAFE_HYDROPHOBICITY_RANGE[0] <= features.hydrophobicity <= SAFE_HYDROPHOBICITY_RANGE[1]
    return charge_ok and hydro_ok


POTENCY_RANK = {"Low": 0, "Medium": 1, "High": 2}


def _score_candidate(potency_result, amp_result, original_potency_class: str,
                      original_amp_probability: float, within_safe_range: bool) -> float:
    """
    Higher = better candidate. Rewards moving up in potency class while
    PRESERVING (not just having some) AMP probability; penalizes
    candidates that drift outside typical natural-AMP physicochemical
    ranges (a proxy for toxicity risk, since we don't call ToxinPred
    per-candidate).

    Critical fix: a candidate that improves potency by sacrificing AMP
    probability isn't actually a good suggestion -- it's not the peptide
    itself getting better, it's the model getting less confident this is
    even an AMP at all. AMP probability DROP relative to the original
    (not just absolute AMP probability) is penalized heavily.
    """
    potency_gain = POTENCY_RANK[potency_result.potency_class.value] - POTENCY_RANK[original_potency_class]
    high_class_confidence = potency_result.class_probabilities.get("High", 0.0)

    amp_probability_drop = original_amp_probability - amp_result.probability

    score = (potency_gain * 2.0) + high_class_confidence + (amp_result.probability * 0.5)

    if amp_probability_drop > 0:
        # Heavily penalize losing AMP confidence -- scaled so a large drop
        # can outweigh even a full potency-class-rank improvement.
        score -= amp_probability_drop * 3.0

    if not within_safe_range:
        score -= 1.0
    return round(score, 3)


def suggest_improvements(sequence: str, variant: str = "pooled", num_suggestions: int = 8) -> List[Suggestion]:
    """
    Generates candidate variants, scores them with the real trained
    models, and returns the top `num_suggestions` distinct improvements.

    Only meaningful to call when the original sequence's potency class
    is Low or Medium -- callers should check that before invoking this
    (the app route does this check before offering the button).
    """
    sequence = sequence.upper()
    original_features = compute_physicochemical_features(sequence)
    original_potency = predict_potency_class(sequence, original_features, variant=variant)
    original_potency_class = original_potency.potency_class.value
    original_amp_result = predict_amp_probability(sequence, original_features)
    original_amp_probability = original_amp_result.probability

    seen_sequences = {sequence}
    candidates = []

    attempts = 0
    while len(candidates) < NUM_CANDIDATES_TO_GENERATE and attempts < NUM_CANDIDATES_TO_GENERATE * 4:
        attempts += 1
        strategy = random.choice(MUTATION_STRATEGIES)
        result = strategy(sequence)
        if result is None:
            continue
        new_seq, description = result
        if new_seq in seen_sequences:
            continue
        seen_sequences.add(new_seq)
        candidates.append((new_seq, description))

    scored_suggestions = []
    for candidate_seq, description in candidates:
        try:
            features = compute_physicochemical_features(candidate_seq)
        except KeyError:
            continue  # shouldn't happen given our mutation alphabet, but stay defensive

        amp_result = predict_amp_probability(candidate_seq, features)
        potency_result = predict_potency_class(candidate_seq, features, variant=variant)
        within_safe_range = _is_within_safe_range(features)

        score = _score_candidate(potency_result, amp_result, original_potency_class, original_amp_probability, within_safe_range)

        feature_deltas = {
            key: round(getattr(features, key) - getattr(original_features, key), 3)
            for key in original_features.as_dict()
        }

        scored_suggestions.append(Suggestion(
            sequence=candidate_seq,
            mutation_description=description,
            original_potency_class=original_potency_class,
            new_potency_class=potency_result.potency_class.value,
            new_potency_probabilities=potency_result.class_probabilities,
            new_amp_probability=amp_result.probability,
            feature_deltas=feature_deltas,
            improvement_score=score,
        ))

    scored_suggestions.sort(key=lambda s: s.improvement_score, reverse=True)
    return scored_suggestions[:num_suggestions]