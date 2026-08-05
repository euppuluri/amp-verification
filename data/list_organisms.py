"""
Lists every organism name present in your loaded GRAMPA/DBAASP data, with
row counts and Gram classification -- so you know exactly what string to
pass as --organism to train_potency_model.py, instead of guessing.

Also reports what fraction of rows fall into "unknown" Gram classification
-- GRAM_POSITIVE/GRAM_NEGATIVE in mic_data_loader.py only cover common lab
organisms. If a real data source (especially DBAASP, which tends to have
far more organism name variety than GRAMPA) introduces many organism names
not in those sets, a meaningful chunk of your data silently gets excluded
from any --gram filtered training run. This surfaces that before it bites
you silently.

Run this BEFORE trying to filter by a specific species, especially since
GRAMPA and DBAASP don't necessarily use the same naming convention for the
same organism (e.g. "E. coli" vs "Escherichia coli").

Usage:
    PYTHONPATH=. python -m data.list_organisms
"""

import pandas as pd

from data.mic_data_loader import load_grampa, load_dbaasp, gram_of

# Above this fraction of "unknown" rows, warn that GRAM_POSITIVE/GRAM_NEGATIVE
# in mic_data_loader.py likely needs more organism names added.
UNKNOWN_WARNING_THRESHOLD = 0.15


def main():
    frames = [load_grampa()]
    try:
        frames.append(load_dbaasp())
    except FileNotFoundError:
        print("Note: data/raw/dbaasp.csv not found -- listing GRAMPA organisms only.\n")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=["organism"])

    combined["gram"] = combined["organism"].apply(gram_of)
    counts = combined["organism"].value_counts()

    print(f"{'Organism':40s} {'Rows':>8s}  {'Gram':10s}")
    print("-" * 62)
    for organism, count in counts.items():
        print(f"{organism:40s} {count:8d}  {gram_of(organism):10s}")

    print(f"\n{len(counts)} distinct organism names, {len(combined)} total rows.")
    print("\nTo filter training to one species, use e.g.:")
    print('  PYTHONPATH=. python -m train.train_potency_model --organism "coli"')
    print("(substring match by default -- matches 'E. coli' and 'Escherichia coli' alike)")

    # --- Organism coverage summary ---
    print("\n" + "=" * 62)
    print("GRAM CLASSIFICATION COVERAGE")
    print("=" * 62)
    gram_counts = combined["gram"].value_counts()
    total_rows = len(combined)
    for gram_type in ["positive", "negative", "unknown"]:
        count = gram_counts.get(gram_type, 0)
        pct = 100 * count / total_rows if total_rows else 0
        print(f"  {gram_type:10s}: {count:6d} rows ({pct:5.1f}%)")

    unknown_frac = gram_counts.get("unknown", 0) / total_rows if total_rows else 0
    if unknown_frac > UNKNOWN_WARNING_THRESHOLD:
        unknown_organisms = (
            combined[combined["gram"] == "unknown"]["organism"]
            .value_counts()
            .head(10)
        )
        print(f"\nWARNING: {unknown_frac:.0%} of rows have unrecognized organisms")
        print(f"(over the {UNKNOWN_WARNING_THRESHOLD:.0%} threshold). These are silently")
        print("excluded from any --gram positive/negative filtered training run.")
        print("\nTop unrecognized organisms by row count (candidates to add to")
        print("GRAM_POSITIVE/GRAM_NEGATIVE in data/mic_data_loader.py):")
        for organism, count in unknown_organisms.items():
            print(f"  {organism:40s} {count:6d} rows")
    else:
        print(f"\nOK: only {unknown_frac:.0%} of rows are unclassified (under the "
              f"{UNKNOWN_WARNING_THRESHOLD:.0%} warning threshold).")


if __name__ == "__main__":
    main()