"""
Fetches actual amino acid sequences from UniProt for the CAMPR3
experimentally-validated family index (exported from
https://camp3.bicnirrh.res.in/dbSearch/famSearch.php as a CSV with columns:
Family, GI, UniProt, Name), and writes them to data/raw/campr3.fasta.

CAMPR3's own site only gives you this identifier index, not sequences
directly -- this script uses UniProt's real REST API to turn the UniProt
accessions in that index into actual sequences.

UniProt API reference (current, confirmed against their docs):
    https://rest.uniprot.org/uniprotkb/stream?query=...&format=fasta
Batch query syntax: query=accession:P15450 OR accession:P81463 OR ...

Usage:
    pip install requests
    python data/fetch_campr3_from_uniprot.py data/raw/campr3_index.csv
"""

import csv
import re
import sys
import time
from pathlib import Path

import requests

UNIPROT_STREAM_URL = "https://rest.uniprot.org/uniprotkb/stream"
BATCH_SIZE = 100          # keep query URLs a reasonable length
REQUEST_DELAY_SECONDS = 1  # be polite to UniProt's servers between batches

RAW_DIR = Path(__file__).parent / "raw"
OUT_PATH = RAW_DIR / "campr3.fasta"

ACCESSION_PATTERN = re.compile(r"[A-Z][0-9][A-Z0-9]{3}[0-9](?:-\d+)?|[OPQ][0-9][A-Z0-9]{3}[0-9](?:-\d+)?")


def extract_accessions(cell: str):
    """
    Handles messy cell contents: single accessions, comma-separated multiple
    accessions (e.g. "P34034 , Q53446"), and blank cells. Returns a list
    (possibly empty) of clean accession strings.
    """
    if not cell or not cell.strip():
        return []
    # Split on commas first (handles "P34034 , Q53446"), then re-extract
    # with the regex as a safety net against any other stray formatting.
    candidates = [c.strip() for c in cell.split(",")]
    accessions = []
    for c in candidates:
        match = ACCESSION_PATTERN.fullmatch(c.strip())
        if match:
            accessions.append(match.group(0))
    return accessions


def load_accessions_from_csv(csv_path: str):
    accessions = []
    skipped_rows = 0

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            found = extract_accessions(row.get("UniProt", ""))
            if found:
                accessions.extend(found)
            else:
                skipped_rows += 1

    # Dedupe while preserving order (some accessions may repeat across rows)
    unique_accessions = list(dict.fromkeys(accessions))
    return unique_accessions, skipped_rows


def batched(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def fetch_batch(accessions):
    query = " OR ".join(f"accession:{acc}" for acc in accessions)
    response = requests.get(
        UNIPROT_STREAM_URL,
        params={"query": query, "format": "fasta"},
        timeout=60,
    )
    response.raise_for_status()
    return response.text


def main():
    if len(sys.argv) != 2:
        print("Usage: python data/fetch_campr3_from_uniprot.py <path_to_exported_csv>")
        sys.exit(1)

    csv_path = sys.argv[1]
    accessions, skipped_rows = load_accessions_from_csv(csv_path)

    print(f"Found {len(accessions)} unique UniProt accessions "
          f"({skipped_rows} rows had no usable accession and were skipped).")

    all_fasta_text = []
    fetched_accessions = set()

    batches = list(batched(accessions, BATCH_SIZE))
    for i, batch in enumerate(batches, 1):
        print(f"Fetching batch {i}/{len(batches)} ({len(batch)} accessions)...")
        try:
            fasta_text = fetch_batch(batch)
        except requests.RequestException as e:
            print(f"  Batch {i} failed: {e}")
            continue

        all_fasta_text.append(fasta_text)
        # Track which accessions actually came back, to report any misses
        for line in fasta_text.splitlines():
            if line.startswith(">"):
                for acc in batch:
                    if acc in line:
                        fetched_accessions.add(acc)

        if i < len(batches):
            time.sleep(REQUEST_DELAY_SECONDS)

    RAW_DIR.mkdir(exist_ok=True)
    OUT_PATH.write_text("\n".join(all_fasta_text))

    missing = set(accessions) - fetched_accessions
    print(f"\nWrote sequences for {len(fetched_accessions)} / {len(accessions)} accessions to {OUT_PATH}")
    if missing:
        print(f"{len(missing)} accessions were not found in UniProt (obsolete/merged entries, likely):")
        for acc in sorted(missing):
            print(f"  {acc}")


if __name__ == "__main__":
    main()