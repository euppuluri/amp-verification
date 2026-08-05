"""
Generates FAKE but realistically-shaped data so you can run the training
scripts end-to-end before downloading the real datasets. A model trained
on this data is NOT scientifically meaningful — just a plumbing test.
"""

import random
from pathlib import Path

random.seed(42)

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
CATIONIC_HELIX_BIASED = "KRLIWFAG"

RAW_DIR = Path(__file__).parent / "raw"
RAW_DIR.mkdir(exist_ok=True)


def _random_sequence(length: int, alphabet: str = AMINO_ACIDS) -> str:
    return "".join(random.choice(alphabet) for _ in range(length))


def generate_mic_csvs(n_peptides: int = 500):
    """Fake data matching GRAMPA's real schema (bacterium,sequence,strain,value)
    and a placeholder DBAASP-style schema."""
    grampa_rows = ["bacterium,sequence,strain,value"]
    dbaasp_rows = ["SEQUENCE,MIC_VALUE,TARGET"]

    for _ in range(n_peptides):
        length = random.randint(10, 30)
        seq = _random_sequence(length, CATIONIC_HELIX_BIASED)
        mic = round(random.lognormvariate(mu=2.0, sigma=1.2), 2)
        organism = random.choice(["E. coli", "S. aureus", "P. aeruginosa"])
        strain = "ATCC25922"

        grampa_rows.append(f"{organism},{seq},{strain},{mic}")
        dbaasp_rows.append(f"{seq},{mic},{organism}")

    (RAW_DIR / "grampa.csv").write_text("\n".join(grampa_rows))
    (RAW_DIR / "dbaasp.csv").write_text("\n".join(dbaasp_rows))
    print(f"Wrote {n_peptides} fake rows each to grampa.csv and dbaasp.csv")


def generate_amp_reference_fastas(n_positive: int = 300, n_negative: int = 300):
    """Fake CAMPR3/CAMPR4/APD3 positives + a negative set."""

    def write_fasta(path, sequences, prefix):
        lines = []
        for i, seq in enumerate(sequences):
            lines.append(f">{prefix}_{i}")
            lines.append(seq)
        path.write_text("\n".join(lines))

    campr3_seqs = [_random_sequence(random.randint(10, 30), CATIONIC_HELIX_BIASED) for _ in range(n_positive // 3)]
    campr4_seqs = [_random_sequence(random.randint(10, 30), CATIONIC_HELIX_BIASED) for _ in range(n_positive // 3)]
    apd3_seqs = [_random_sequence(random.randint(10, 30), CATIONIC_HELIX_BIASED) for _ in range(n_positive // 3)]
    negative_seqs = [_random_sequence(random.randint(10, 30)) for _ in range(n_negative)]

    write_fasta(RAW_DIR / "campr3.fasta", campr3_seqs, "campr3")
    write_fasta(RAW_DIR / "campr4.fasta", campr4_seqs, "campr4")
    write_fasta(RAW_DIR / "apd3.fasta", apd3_seqs, "apd3")
    write_fasta(RAW_DIR / "negatives.fasta", negative_seqs, "neg")

    print(f"Wrote fake FASTA files: {n_positive} positives (split across 3 files), {n_negative} negatives")


if __name__ == "__main__":
    generate_mic_csvs()
    generate_amp_reference_fastas()
    print(f"\nAll fake raw data written to: {RAW_DIR}")
    print("You can now run:")
    print("  PYTHONPATH=. python -m train.train_potency_model")
    print("  PYTHONPATH=. python -m train.train_amp_classifier")