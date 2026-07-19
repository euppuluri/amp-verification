"""
Wraps ToxinPred 3.0 to produce a numeric toxicity score.

Confirmed install/usage from https://github.com/raghavagps/toxinpred3 :

    pip install toxinpred3
    toxinpred3 -i input.fasta -o output.csv -t 0.38 -m 2 -d 2

Flags:
    -t  threshold, 0-1, default 0.38
    -m  model: 1 = ML only, 2 = Hybrid (ML + MERCI), default 2
    -d  display: 1 = toxin peptides only, 2 = ALL peptides, default 1
        (-d 2 is required here so a non-toxic query still shows up)

Output CSV columns: Subject, ML Score, MERCI Score Pos, MERCI Score Neg,
Hybrid Score, Prediction, PPV. We use 'Hybrid Score' since we run model 2.
"""

from dataclasses import dataclass
import subprocess
import tempfile
import csv
from pathlib import Path


@dataclass
class ToxicityResult:
    toxicity_score: float
    is_toxic: bool


def predict_toxicity(sequence: str) -> ToxicityResult:
    with tempfile.TemporaryDirectory() as tmp:
        fasta_path = Path(tmp) / "input.fasta"
        out_path = Path(tmp) / "output.csv"
        fasta_path.write_text(f">query\n{sequence}\n")

        subprocess.run(
            [
                "toxinpred3",
                "-i", str(fasta_path),
                "-o", str(out_path),
                "-t", "0.38",
                "-m", "2",
                "-d", "2",
            ],
            check=True,
            capture_output=True,
        )

        with open(out_path) as f:
            row = next(csv.DictReader(f))

        return ToxicityResult(
            toxicity_score=float(row["Hybrid Score"]),
            is_toxic=row["Prediction"].strip().lower() == "toxin",
        )