"""
Generates data/raw/negatives.fasta -- non-AMP peptide fragments for
training the AMP classifier, following standard practice in AMP
classifier literature (e.g. AMPScanner, iAMPpred): random fragments cut
from reviewed (Swiss-Prot) proteins that have no antimicrobial-related
annotation, length-matched to typical AMP length ranges.

Two layers of exclusion, since getting this wrong means training on
mislabeled "negatives" that are secretly antimicrobial:
  1. Query-level: excludes UniProt's own "Antimicrobial" keyword (KW-0929)
     at the API level.
  2. Text-level safety net: re-checks each protein's description against a
     list of AMP-adjacent terms (defensin, cecropin, magainin, bacteriocin,
     cathelicidin, antibiotic, toxin, etc.) and skips any match, in case
     the keyword-level exclusion missed something (a protein described as
     antimicrobial without that exact keyword attached, for instance).

UniProt API reference (confirmed against their current docs):
    https://rest.uniprot.org/uniprotkb/search?query=...&format=fasta&size=500
Pagination via the 'Link' response header (standard UniProt cursor pattern).

Usage:
    pip install requests
    python data/generate_negatives.py --count 500
"""

import argparse
import random
import re
import sys
from pathlib import Path

import requests

UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
PAGE_SIZE = 500

# Standard 20 amino acids only -- our physicochemical feature formulas
# don't cover ambiguous/modified residues (X, B, Z, U, O), so fragments
# containing them are skipped rather than crashing training later.
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

# Safety-net keyword list: skip any protein whose description mentions
# these, even though the query already excludes UniProt's "Antimicrobial"
# keyword at the API level.
SUSPICIOUS_TERMS = [
    "antimicrob", "antibiotic", "defensin", "cecropin", "magainin",
    "bacteriocin", "cathelicidin", "antifungal", "antiviral peptide",
    "bacteriolytic", "antibacterial",
]

RAW_DIR = Path(__file__).parent / "raw"
OUT_PATH = RAW_DIR / "negatives.fasta"


def is_suspicious(description: str) -> bool:
    desc_lower = description.lower()
    return any(term in desc_lower for term in SUSPICIOUS_TERMS)


def parse_fasta_text(fasta_text: str):
    """Yields (description, sequence) for each record in raw FASTA text."""
    description = None
    seq_lines = []
    for line in fasta_text.splitlines():
        if line.startswith(">"):
            if description is not None:
                yield description, "".join(seq_lines)
            description = line[1:]
            seq_lines = []
        else:
            seq_lines.append(line.strip())
    if description is not None:
        yield description, "".join(seq_lines)


def fetch_background_proteins(target_count: int, min_protein_len: int = 80, max_pages: int = 20):
    """
    Fetches reviewed, non-antimicrobial-annotated proteins from UniProt,
    paginating until target_count clean proteins are collected or
    max_pages is hit (safety cap so a bad query can't loop forever).
    """
    query = (
        f"reviewed:true AND NOT keyword:KW-0929 "
        f"AND length:[{min_protein_len} TO 1000]"
    )
    params = {"query": query, "format": "fasta", "size": PAGE_SIZE}
    url = UNIPROT_SEARCH_URL

    collected = []
    pages_fetched = 0

    while url and len(collected) < target_count and pages_fetched < max_pages:
        response = requests.get(url, params=params if pages_fetched == 0 else None, timeout=60)
        response.raise_for_status()
        pages_fetched += 1

        for description, sequence in parse_fasta_text(response.text):
            if is_suspicious(description):
                continue
            if not sequence or not set(sequence.upper()).issubset(VALID_AA):
                continue
            collected.append((description, sequence))

        # UniProt cursor pagination: next page URL is in the Link header
        next_url = None
        link_header = response.headers.get("Link", "")
        match = re.search(r'<([^>]+)>;\s*rel="next"', link_header)
        if match:
            next_url = match.group(1)
        url = next_url
        params = None  # next_url already has query params embedded

        print(f"  Fetched page {pages_fetched}: {len(collected)} clean background proteins so far...")

    return collected


def extract_fragments(sequence: str, min_len: int, max_len: int, fragments_per_protein: int):
    """Cuts a small number of non-overlapping random-length fragments from one protein."""
    fragments = []
    used_ranges = []

    attempts = 0
    while len(fragments) < fragments_per_protein and attempts < fragments_per_protein * 5:
        attempts += 1
        frag_len = random.randint(min_len, max_len)
        if frag_len >= len(sequence):
            continue
        start = random.randint(0, len(sequence) - frag_len)
        end = start + frag_len

        if any(start < e and end > s for s, e in used_ranges):
            continue  # overlaps a fragment we already took from this protein

        fragment = sequence[start:end]
        if set(fragment.upper()).issubset(VALID_AA):
            fragments.append(fragment)
            used_ranges.append((start, end))

    return fragments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=500,
                         help="Target number of negative peptide fragments to generate.")
    parser.add_argument("--min-len", type=int, default=5, help="Minimum fragment length.")
    parser.add_argument("--max-len", type=int, default=60, help="Maximum fragment length.")
    parser.add_argument("--fragments-per-protein", type=int, default=2,
                         help="How many fragments to cut from each background protein.")
    args = parser.parse_args()

    proteins_needed = max(10, args.count // args.fragments_per_protein + 5)
    print(f"Fetching ~{proteins_needed} clean background proteins from UniProt...")

    try:
        proteins = fetch_background_proteins(target_count=proteins_needed)
    except requests.RequestException as e:
        print(f"UniProt request failed: {e}")
        sys.exit(1)

    print(f"\nGot {len(proteins)} usable background proteins. Cutting fragments...")

    fasta_lines = []
    total_fragments = 0

    for i, (description, sequence) in enumerate(proteins):
        if total_fragments >= args.count:
            break
        fragments = extract_fragments(sequence, args.min_len, args.max_len, args.fragments_per_protein)
        for j, fragment in enumerate(fragments):
            if total_fragments >= args.count:
                break
            fasta_lines.append(f">neg_{total_fragments}_from_{description[:40]}")
            fasta_lines.append(fragment)
            total_fragments += 1

    if total_fragments < args.count:
        print(f"WARNING: only generated {total_fragments}/{args.count} requested fragments "
              f"-- try increasing --fragments-per-protein or check UniProt query results above.")

    RAW_DIR.mkdir(exist_ok=True)
    OUT_PATH.write_text("\n".join(fasta_lines))
    print(f"\nWrote {total_fragments} negative fragments to {OUT_PATH}")


if __name__ == "__main__":
    main()