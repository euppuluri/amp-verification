"""
Retrains everything in one command, against whatever real data currently
exists in data/raw/, and reports a summary at the end.

This does NOT fetch/download data for you -- it assumes you've already:
  1. Downloaded GRAMPA to data/raw/grampa.csv
     https://github.com/zswitten/Antimicrobial-Peptides/raw/refs/heads/master/data/grampa.csv
  2. Generated DBAASP via: python data/fetch_dbaasp.py (writes data/raw/dbaasp.csv)
  3. Exported CAMPR3/CAMPR4 sequences to data/raw/campr3.fasta and campr4.fasta
     https://camp3.bicnirrh.res.in/  https://camp.bicnirrh.res.in/
  4. Downloaded APD6 (formerly APD3) bulk category FASTA files from
     https://aps.unmc.edu/downloads and concatenated them into
     data/raw/apd3.fasta
  5. Put together data/raw/negatives.fasta (e.g. random non-AMP UniProt
     fragments -- there's no single canonical negative set, this is on you)

Missing files are handled gracefully where the underlying loaders already
support it (DBAASP is optional), but AMP classifier training needs
campr3.fasta, campr4.fasta, and negatives.fasta to exist.

Usage:
    PYTHONPATH=. python -m train.retrain_all
"""

import subprocess
import sys
from pathlib import Path

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"

REQUIRED_FOR_AMP = ["campr3.fasta", "campr4.fasta", "negatives.fasta", "apd3.fasta"]
REQUIRED_FOR_POTENCY = ["grampa.csv"]


def check_files(filenames):
    missing = [f for f in filenames if not (RAW_DIR / f).exists()]
    return missing


def run(cmd):
    print(f"\n{'=' * 60}\n$ {' '.join(cmd)}\n{'=' * 60}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"FAILED: {' '.join(cmd)}")
        return False
    return True


def main():
    print("Checking for required data files...")
    missing_amp = check_files(REQUIRED_FOR_AMP)
    missing_potency = check_files(REQUIRED_FOR_POTENCY)

    if missing_amp:
        print(f"Missing for AMP classifier training: {missing_amp}")
        print("AMP classifier + external validation will be skipped.")
    if missing_potency:
        print(f"Missing for potency training: {missing_potency}")
        print("Potency model training will be skipped.")

    results = {}

    if not missing_amp:
        results["amp_classifier"] = run([sys.executable, "-m", "train.train_amp_classifier"])
        results["external_validation"] = run([sys.executable, "-m", "train.validate_external"])

    if not missing_potency:
        results["potency_pooled"] = run([sys.executable, "-m", "train.train_potency_model"])
        results["potency_gram_negative"] = run(
            [sys.executable, "-m", "train.train_potency_model", "--gram", "negative"])
        results["potency_gram_positive"] = run(
            [sys.executable, "-m", "train.train_potency_model", "--gram", "positive"])

    print(f"\n{'=' * 60}\nSUMMARY\n{'=' * 60}")
    if not results:
        print("Nothing ran -- no required data files were present. See messages above.")
    for step, success in results.items():
        status = "OK" if success else "FAILED"
        print(f"  {step:30s} {status}")


if __name__ == "__main__":
    main()