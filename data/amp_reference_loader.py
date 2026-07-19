"""
Loads AMP/non-AMP reference sequences from CAMPR3, CAMPR4, and APD3.

Homepages (manual export needed — no bulk API):
    CAMPR3: https://camp3.bicnirrh.res.in/
    CAMPR4: https://camp.bicnirrh.res.in/
    APD3:   https://aps.unmc.edu/

Place exports at:
    data/raw/campr3.fasta
    data/raw/campr4.fasta
    data/raw/apd3.fasta
    data/raw/negatives.fasta   (e.g. random non-AMP UniProt fragments)
"""

import pandas as pd


def _parse_fasta(path: str) -> list[str]:
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
            "label": label,
        }
    )


def build_amp_classifier_dataset() -> pd.DataFrame:
    positives = pd.concat(
        [
            _load_fasta_as_df("data/raw/campr3.fasta", label=1),
            _load_fasta_as_df("data/raw/campr4.fasta", label=1),
            _load_fasta_as_df("data/raw/apd3.fasta", label=1),
        ],
        ignore_index=True,
    ).drop_duplicates(subset="sequence")

    negatives = _load_fasta_as_df("data/raw/negatives.fasta", label=0)

    combined = pd.concat([positives, negatives], ignore_index=True)
    combined = combined[combined["sequence"].str.len().between(5, 60)]
    return combined