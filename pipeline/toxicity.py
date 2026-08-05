"""
Wraps ToxinPred 3.0 to produce a numeric toxicity score.

Confirmed install/usage from https://github.com/raghavagps/toxinpred3,
and verified directly against the real installed CLI's --help output:

    pip install toxinpred3
    toxinpred3 -i input.fasta -o output.csv -t 0.38 -m 2 -d 2

Flags (confirmed against real `toxinpred3 --help` output):
    -t  threshold, 0-1, default 0.38
    -m  model: 1 = AAC & DPC based ET, 2 = Hybrid, default 1 (we use 2)
    -d  display: 1 = toxin peptides only, 2 = ALL peptides, default 2

Output CSV columns (confirmed against real output): Subject, ML Score,
MERCI Score Pos, MERCI Score Neg, Hybrid Score, Prediction, PPV. We use
'Hybrid Score' since we run model 2.

IMPORTANT -- environment isolation:
ToxinPred 3.0's pickled model requires an older scikit-learn/numpy
combination than this project's own training scripts use. Install
ToxinPred in a SEPARATE venv (see README), then point TOXINPRED_BIN at
that venv's toxinpred3 executable, e.g.:

    export TOXINPRED_BIN="/path/to/venv-toxinpred/bin/toxinpred3"

If TOXINPRED_BIN isn't set, this falls back to just "toxinpred3" (i.e.
whatever's on PATH in the currently active environment).

IMPORTANT -- single-sequence bug:
ToxinPred 3.0 throws a numpy AxisError when given exactly one sequence
(confirmed against the real installed CLI). We work around this by
padding every query with a second, harmless dummy sequence, then only
reading back the row matching our actual query, identified by name.
"""

import os
from dataclasses import dataclass
import subprocess
import tempfile
import csv
from pathlib import Path

TOXINPRED_BIN = os.environ.get("TOXINPRED_BIN", "toxinpred3")


@dataclass
class ToxicityResult:
    toxicity_score: float
    is_toxic: bool


def predict_toxicity(sequence: str) -> ToxicityResult:
    with tempfile.TemporaryDirectory() as tmp:
        fasta_path = Path(tmp) / "input.fasta"
        out_path = Path(tmp) / "output.csv"
        fasta_path.write_text(f">query\n{sequence}\n>__padding__\nAAAAAAAAAAAAAAAAAAAA\n")

        try:
            subprocess.run(
                [
                    TOXINPRED_BIN,
                    "-i", str(fasta_path),
                    "-o", str(out_path),
                    "-t", "0.38",
                    "-m", "2",
                    "-d", "2",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"toxinpred3 failed (exit code {e.returncode}).\n"
                f"--- stdout ---\n{e.stdout}\n"
                f"--- stderr ---\n{e.stderr}"
            ) from e

        with open(out_path) as f:
            rows = list(csv.DictReader(f))

        query_rows = [r for r in rows if r["Subject"] == "query"]
        if not query_rows:
            raise RuntimeError(
                f"No 'query' row found in ToxinPred output. Got rows: {rows}"
            )
        row = query_rows[0]

        return ToxicityResult(
            toxicity_score=float(row["Hybrid Score"]),
            is_toxic=row["Prediction"].strip().lower() == "toxin",
        )