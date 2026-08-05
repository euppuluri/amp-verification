"""
Fetches AMP activity data from the DBAASP REST API and writes it to
data/raw/dbaasp.csv in the schema mic_data_loader.py expects.

DBAASP's API is documented at:
    https://github.com/melomcr/dbaasp_api_helper_libraries
Endpoint: https://dbaasp.org/api/v1

This script's exact field-parsing (in the TODO section below) is a best
guess -- DBAASP doesn't publish a detailed schema for the JSON response
body. Run this script, look at the printed raw structure of the first
result, and adjust the TODO section to match what you actually see.

Usage:
    pip install requests
    python data/fetch_dbaasp.py
"""

import requests
import json
import csv
from pathlib import Path

API_BASE = "https://dbaasp.org/api/v1"
RAW_DIR = Path(__file__).parent / "raw"
RAW_DIR.mkdir(exist_ok=True)


def lookup(lookup_type: str):
    """
    Fetch the ID -> name mapping for a given lookup_type.
    Valid types include: target_species, activity_measure, target_group,
    kingdom, synthesis_type, unit, and others (see the API README).
    """
    resp = requests.get(
        API_BASE,
        params={"query": "lookup", "lookup_type": lookup_type, "format": "json"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def search_peptides(target_species_id: int, kingdom_id: int = None):
    """Fetch peptides tested against a given target species."""
    params = {"query": "search", "target_species_id": target_species_id, "format": "json"}
    if kingdom_id is not None:
        params["kingdom_id"] = kingdom_id
    resp = requests.get(API_BASE, params=params, timeout=60)
    resp.raise_for_status()
    return resp.json()


if __name__ == "__main__":
    print("Fetching target species list from DBAASP...")
    species = lookup("target_species")
    print(json.dumps(species, indent=2)[:2000])
    print("\n--> Look through the list above for a species you want (e.g. Escherichia coli), note its ID.")

    species_id = input("\nEnter target_species_id to fetch peptides for: ").strip()
    results = search_peptides(int(species_id))

    print("\n--- Raw structure of the first result (READ THIS before trusting the CSV below) ---")
    first = results[0] if isinstance(results, list) else results
    print(json.dumps(first, indent=2)[:3000])
    print("--- end raw structure ---\n")

    # TODO: once you've seen the real field names printed above, fix these
    # three lines to pull the correct keys (these are placeholder guesses):
    rows = []
    records = results if isinstance(results, list) else results.get("peptides", [])
    for entry in records:
        sequence = entry.get("sequence")
        mic = entry.get("concentration")
        organism = entry.get("targetSpecies", {}).get("name") if isinstance(entry.get("targetSpecies"), dict) else None
        if sequence and mic:
            rows.append((sequence, mic, organism))

    out_path = RAW_DIR / "dbaasp.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["SEQUENCE", "MIC_VALUE", "TARGET"])
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path}")
    if len(rows) == 0:
        print("0 rows means the TODO field names above are wrong -- fix them using the")
        print("raw structure printed earlier, then re-run.")