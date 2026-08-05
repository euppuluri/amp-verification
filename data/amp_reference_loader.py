"""
Loads AMP/non-AMP reference sequences for training and externally
validating the amp_activity classifier.

Homepages (manual export needed — none expose a bulk API):
    CAMPR3: https://camp3.bicnirrh.res.in/
    CAMPR4: https://camp.bicnirrh.res.in/
    APD3:   https://aps.unmc.edu/

Place exports at:
    data/raw/campr3.fasta
    data/raw/campr4.fasta
    data/raw/apd3.fasta
    data/raw/negatives.fasta   (e.g. random non-AMP UniProt fragments,
                                 standard practice for this task)

APD3 is deliberately EXCLUDED from training and reserved as an external
validation set (see train/validate_external.py) — a model can trivially
score well on data it was trained on, so holding out an entire independent
database is a more honest check of whether it generalizes, rather than
just memorizing CAMPR3/CAMPR4.
"""

import pandas as pd
from typing import List


def _parse_fasta(path: str) -> List[str]:
    """Minimal FASTA parser (no biopython dependency needed)."""
    sequences = []
    current = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current:
                    sequences.append("".join(current))
                    current = []
            else:
                current.append(line)
    if current:
        sequences.append("".join(current))
    return sequences


def _load_fasta_as_df(path: str, label: int) -> pd.DataFrame:
    sequences = _parse_fasta(path)
    return pd.DataFrame(
        {
            "sequence": [s.upper() for s in sequences],
            "label": label,  # 1 = AMP, 0 = non-AMP
        }
    )


def build_amp_classifier_dataset() -> pd.DataFrame:
    """
    Training data: CAMPR3 + CAMPR4 positives, plus negatives.
    APD3 is intentionally NOT included here — see build_apd3_validation_set().
    """
    positives = pd.concat(
        [
            _load_fasta_as_df("data/raw/campr3.fasta", label=1),
            _load_fasta_as_df("data/raw/campr4.fasta", label=1),
        ],
        ignore_index=True,
    ).drop_duplicates(subset="sequence")

    negatives = _load_fasta_as_df("data/raw/negatives.fasta", label=0)

    combined = pd.concat([positives, negatives], ignore_index=True)
    combined = combined[combined["sequence"].str.len().between(5, 60)]
    return combined


def build_apd3_validation_set() -> pd.DataFrame:
    """
    External validation set: APD3 peptides, held out entirely from
    training. All labeled 1 (AMP) — APD3 only contains validated AMPs, so
    this checks recall/sensitivity (does the model correctly recognize an
    independent set of known AMPs it has never seen), not full accuracy.
    """
    apd3 = _load_fasta_as_df("data/raw/apd3.fasta", label=1)
    return apd3[apd3["sequence"].str.len().between(5, 60)]