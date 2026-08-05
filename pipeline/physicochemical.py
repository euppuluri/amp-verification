"""
Computes the physicochemical features used both as standalone report items
and as inputs to the AMP-activity and potency-class models.

Implementation notes:
- Everything here is a plain-Python implementation of well-established
  formulas (Kyte-Doolittle hydrophobicity, Eisenberg hydrophobic moment,
  Henderson-Hasselbalch charge, Chou-Fasman helicity propensity). No
  external dependencies needed, and it's compatible back to Python 3.9.
"""

import math
from dataclasses import dataclass
from typing import List, Tuple

# --- Reference scales (standard, widely published values) ---

KYTE_DOOLITTLE = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

EISENBERG = {
    "A": 0.62, "R": -2.53, "N": -0.78, "D": -0.90, "C": 0.29,
    "Q": -0.85, "E": -0.74, "G": 0.48, "H": -0.40, "I": 1.38,
    "L": 1.06, "K": -1.50, "M": 0.64, "F": 1.19, "P": 0.12,
    "S": -0.18, "T": -0.05, "W": 0.81, "Y": 0.26, "V": 1.08,
}

CHOU_FASMAN_HELIX = {
    "A": 1.42, "R": 0.98, "N": 0.67, "D": 1.01, "C": 0.70,
    "Q": 1.11, "E": 1.51, "G": 0.57, "H": 1.00, "I": 1.08,
    "L": 1.21, "K": 1.16, "M": 1.45, "F": 1.13, "P": 0.57,
    "S": 0.77, "T": 0.83, "W": 1.08, "Y": 0.69, "V": 1.06,
}

PKA_SIDE_CHAINS = {
    "D": 3.65, "E": 4.25, "C": 8.3, "Y": 10.1,
    "H": 6.0, "K": 10.5, "R": 12.5,
}
N_TERM_PKA = 9.0
C_TERM_PKA = 2.0

# --- Protease cleavage site rules (simplified specificity) ---
# Trypsin: cleaves after K/R, unless followed by Proline (blocks cleavage)
# Chymotrypsin: cleaves after F/Y/W (aromatic), unless followed by Proline
# Elastase: cleaves after small/aliphatic A/G/V (and sometimes L), unless
#           followed by Proline
# This is a simplified rule set (real protease specificity is more nuanced
# and context-dependent) but captures the dominant, well-established
# cleavage preferences used as a standard first-pass stability heuristic.
TRYPSIN_SITES = set("KR")
CHYMOTRYPSIN_SITES = set("FYW")
ELASTASE_SITES = set("AGV")

# --- Residue classification, used for the sequence-tile visualization ---

RESIDUE_CLASS = {
    "K": "basic", "R": "basic", "H": "basic",
    "D": "acidic", "E": "acidic",
    "A": "hydrophobic", "V": "hydrophobic", "L": "hydrophobic", "I": "hydrophobic",
    "M": "hydrophobic", "F": "hydrophobic", "W": "hydrophobic", "P": "hydrophobic",
    "S": "polar", "T": "polar", "N": "polar", "Q": "polar", "Y": "polar", "C": "polar",
    "G": "glycine",
}


@dataclass
class PhysicochemicalFeatures:
    length: int
    net_charge: float
    hydrophobicity: float          # mean Kyte-Doolittle score
    hydrophobic_moment: float      # amphipathicity, assuming an alpha-helix
    helicity_score: float          # 0-1, mean normalized Chou-Fasman propensity
    aggregation_propensity: float  # heuristic: hydrophobic-run based score
    isoelectric_point: float
    protease_stability: float      # 0-1, higher = fewer predicted cleavage sites

    def as_dict(self) -> dict:
        return self.__dict__


def _net_charge_at_ph(sequence: str, ph: float = 7.4) -> float:
    """Henderson-Hasselbalch based net charge estimate."""
    charge = 0.0

    # N-terminus (positive when protonated)
    charge += 1 / (1 + 10 ** (ph - N_TERM_PKA))
    # C-terminus (negative when deprotonated)
    charge -= 1 / (1 + 10 ** (C_TERM_PKA - ph))

    for aa in sequence:
        if aa in ("D", "E", "C", "Y"):
            pka = PKA_SIDE_CHAINS[aa]
            charge -= 1 / (1 + 10 ** (pka - ph))  # acidic: negative when deprotonated
        elif aa in ("K", "R", "H"):
            pka = PKA_SIDE_CHAINS[aa]
            charge += 1 / (1 + 10 ** (ph - pka))  # basic: positive when protonated

    return round(charge, 3)


def _hydrophobicity(sequence: str) -> float:
    return round(sum(KYTE_DOOLITTLE[aa] for aa in sequence) / len(sequence), 3)


def _hydrophobic_moment(sequence: str, angle_deg: float = 100.0) -> float:
    """
    Eisenberg hydrophobic moment, assuming an idealized alpha-helix
    (100 degrees per residue). Higher = more amphipathic.
    """
    angle_rad = math.radians(angle_deg)
    sum_cos, sum_sin = 0.0, 0.0
    for i, aa in enumerate(sequence):
        h = EISENBERG[aa]
        sum_cos += h * math.cos(i * angle_rad)
        sum_sin += h * math.sin(i * angle_rad)
    moment = math.sqrt(sum_cos ** 2 + sum_sin ** 2) / len(sequence)
    return round(moment, 3)


def _helicity_score(sequence: str) -> float:
    """Mean Chou-Fasman helix propensity, normalized to roughly 0-1."""
    raw_mean = sum(CHOU_FASMAN_HELIX[aa] for aa in sequence) / len(sequence)
    normalized = (raw_mean - 0.5) / (1.5 - 0.5)
    return round(max(0.0, min(1.0, normalized)), 3)


def _aggregation_propensity(sequence: str) -> float:
    """
    Heuristic proxy: consecutive stretches of hydrophobic, beta-sheet-prone
    residues (I, V, F, Y, W, L) are the classic driver of aggregation.
    Score = fraction of residues sitting inside a hydrophobic run of >= 3.
    This is a simplification, not a substitute for a dedicated tool like
    TANGO or AGGRESCAN — flagged here as a proxy, not a hard prediction.
    """
    agg_prone = set("IVFYWL")
    run_length = 0
    flagged_residues = 0
    for aa in sequence:
        if aa in agg_prone:
            run_length += 1
        else:
            if run_length >= 3:
                flagged_residues += run_length
            run_length = 0
    if run_length >= 3:
        flagged_residues += run_length
    return round(flagged_residues / len(sequence), 3)


def _protease_stability(sequence: str) -> float:
    """
    Fraction of peptide bonds NOT predicted to be protease-susceptible,
    using simplified trypsin/chymotrypsin/elastase cleavage rules.
    Returns 0-1, where 1.0 = no predicted cleavage sites (more stable)
    and lower values = more predicted cleavage sites (less stable).

    This is a coarse first-pass proxy, not a substitute for a dedicated
    tool or actual serum-stability assay — flagged as a heuristic.
    """
    if len(sequence) < 2:
        return 1.0

    cleavage_sites = 0
    total_bonds = len(sequence) - 1

    for i in range(total_bonds):
        residue = sequence[i]
        next_residue = sequence[i + 1]
        if next_residue == "P":
            continue  # Proline blocks cleavage for all three rules
        if residue in TRYPSIN_SITES or residue in CHYMOTRYPSIN_SITES or residue in ELASTASE_SITES:
            cleavage_sites += 1

    return round(1.0 - (cleavage_sites / total_bonds), 3)


def _isoelectric_point(sequence: str) -> float:
    """Binary search for the pH where net charge crosses zero."""
    low, high = 0.0, 14.0
    for _ in range(50):
        mid = (low + high) / 2
        charge = _net_charge_at_ph(sequence, mid)
        if charge > 0:
            low = mid
        else:
            high = mid
    return round((low + high) / 2, 2)


def compute_physicochemical_features(sequence: str) -> PhysicochemicalFeatures:
    sequence = sequence.upper()
    return PhysicochemicalFeatures(
        length=len(sequence),
        net_charge=_net_charge_at_ph(sequence),
        hydrophobicity=_hydrophobicity(sequence),
        hydrophobic_moment=_hydrophobic_moment(sequence),
        helicity_score=_helicity_score(sequence),
        aggregation_propensity=_aggregation_propensity(sequence),
        isoelectric_point=_isoelectric_point(sequence),
        protease_stability=_protease_stability(sequence),
    )


def classify_sequence(sequence: str) -> List[Tuple[str, str]]:
    """Returns [(residue, css_class), ...] for rendering colored sequence tiles."""
    return [(aa, RESIDUE_CLASS.get(aa, "polar")) for aa in sequence.upper()]