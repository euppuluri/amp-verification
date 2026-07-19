"""Basic input validation for peptide sequences."""

from typing import Optional, Tuple

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

MIN_LEN = 5
MAX_LEN = 60  # beyond this, "AMP" framing / MIC databases stop being representative


def validate_sequence(sequence: str) -> Tuple[bool, Optional[str]]:
    if not sequence:
        return False, "Please enter a peptide sequence."

    if not (MIN_LEN <= len(sequence) <= MAX_LEN):
        return False, f"Sequence length must be between {MIN_LEN} and {MAX_LEN} residues."

    invalid_chars = set(sequence) - VALID_AA
    if invalid_chars:
        return False, f"Invalid characters found: {', '.join(sorted(invalid_chars))}"

    return True, None