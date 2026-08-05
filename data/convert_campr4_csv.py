"""
Converts a CAMPR4 CSV export (columns: Camp_ID, Title, Source_Organism,
Taxonomy, Seqence, Length, Pubmed_id, Activity, Validation, Modifications,
Target) into data/raw/campr4.fasta, keeping ONLY experimentally validated
entries -- same principle applied to CAMPR3 (see amp_reference_loader.py
docstring): predicted/signature-based entries aren't lab-confirmed, so
training on them risks label noise and circularity.

The Validation column has inconsistent casing/whitespace in the real
export ("Experimentally validated" vs "Experimentally Validated" vs
"Predicted " with a trailing space, etc.) -- this script normalizes
before matching so none of those variants get missed or double-counted.

File encoding: the raw export is cp1252, not UTF-8 (it'll fail to read
with pandas' default encoding otherwise).

Sequences using non-standard notation (e.g. synthetic lipopeptides like
"C8R-C8R2KK") are written to the FASTA as-is -- they'll be automatically
skipped later by train_amp_classifier.py's existing defensive handling
for non-standard residues, with a printed count, rather than needing
separate filtering here.

Usage:
    python data/convert_campr4_csv.py /path/to/campr4_data.csv
"""

import sys
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).parent / "raw"
OUT_PATH = RAW_DIR / "campr4.fasta"


def main():
    if len(sys.argv) != 2:
        print("Usage: python data/convert_campr4_csv.py /path/to/campr4_data.csv")
        sys.exit(1)

    csv_path = sys.argv[1]
    df = pd.read_csv(csv_path, encoding="cp1252")
    df.columns = df.columns.str.strip()  # the real export has "  Camp_ID" with leading spaces

    print(f"Total rows in export: {len(df)}")

    # Normalize Validation column: strip whitespace, lowercase, so
    # "Experimentally validated", "Experimentally Validated", and any
    # trailing-space variants all match consistently.
    validation_norm = df["Validation"].astype(str).str.strip().str.lower()
    is_validated = validation_norm.str.startswith("experimentally validated")

    validated_df = df[is_validated]
    print(f"Experimentally validated rows: {len(validated_df)}")
    print(f"(Excluded {len(df) - len(validated_df)} predicted/signature-based/other rows)")

    validated_df = validated_df.dropna(subset=["Seqence"])
    validated_df = validated_df[validated_df["Seqence"].astype(str).str.strip() != ""]

    fasta_lines = []
    for row in validated_df.itertuples(index=False):
        camp_id = getattr(row, "Camp_ID", "unknown")
        sequence = str(getattr(row, "Seqence")).strip()
        fasta_lines.append(f">{camp_id}")
        fasta_lines.append(sequence)

    RAW_DIR.mkdir(exist_ok=True)
    OUT_PATH.write_text("\n".join(fasta_lines))

    print(f"\nWrote {len(validated_df)} sequences to {OUT_PATH}")
    print("(Sequences with non-standard notation, e.g. synthetic lipopeptides,")
    print(" are included here but will be auto-skipped during feature")
    print(" computation later -- train_amp_classifier.py already handles this.)")


if __name__ == "__main__":
    main()