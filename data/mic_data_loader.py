"""
Loads and harmonizes MIC-labeled peptide data from GRAMPA and DBAASP for
training the potency classifier.

GRAMPA download (no account needed, browser download works — automated
fetching is blocked by GitHub's robots.txt):
    https://github.com/zswitten/Antimicrobial-Peptides/raw/refs/heads/master/data/grampa.csv
Save it to: data/raw/grampa.csv

Real GRAMPA columns (confirmed from the source repo's README):
    bacterium, sequence, strain, value, database, url_source, modifications,
    unit (always 'uM'), is_modified, has_unusual_modification,
    has_cterminal_amidation, datasource_has_modifications
GRAMPA contains 6,760 unique peptides, 51,345 total MIC measurements.

DBAASP has no bulk CSV endpoint — see data/fetch_dbaasp.py for a script
that pulls it via DBAASP's REST API instead. DBAASP is OPTIONAL: if
data/raw/dbaasp.csv doesn't exist yet, this loader just trains on GRAMPA.

Why organism/Gram filtering matters: mixing MIC values across many
different bacteria (some intrinsically much easier or harder to kill)
means "potency" partly reflects which organism a peptide happened to be
tested against, not the peptide's real activity. Filtering to a single
organism, or at least stratifying by Gram type, removes that confound.
"""

import pandas as pd
import numpy as np

# Gram classification for organisms commonly seen in GRAMPA/DBAASP.
# Not exhaustive — anything not listed falls into 'unknown' and is
# excluded from Gram-stratified training (rather than silently
# mis-assigned to the wrong bucket).
GRAM_POSITIVE = {
    "s. aureus", "staphylococcus aureus",
    "b. subtilis", "bacillus subtilis",
    "e. faecalis", "enterococcus faecalis",
    "l. monocytogenes", "listeria monocytogenes",
    "s. pyogenes", "streptococcus pyogenes",
    "s. epidermidis", "staphylococcus epidermidis",
    "m. luteus", "micrococcus luteus",
}

GRAM_NEGATIVE = {
    "e. coli", "escherichia coli",
    "p. aeruginosa", "pseudomonas aeruginosa",
    "k. pneumoniae", "klebsiella pneumoniae",
    "a. baumannii", "acinetobacter baumannii",
    "s. typhimurium", "salmonella typhimurium",
    "e. cloacae", "enterobacter cloacae",
    "p. mirabilis", "proteus mirabilis",
}


def gram_of(organism: str) -> str:
    """Returns 'positive', 'negative', or 'unknown' for a given organism name."""
    if not isinstance(organism, str):
        return "unknown"
    name = organism.strip().lower()
    if name in GRAM_POSITIVE:
        return "positive"
    if name in GRAM_NEGATIVE:
        return "negative"
    return "unknown"


def load_grampa(path: str = "data/raw/grampa.csv") -> pd.DataFrame:
    """Returns columns: sequence, mic_um, organism, source='grampa'."""
    df = pd.read_csv(path)
    df = df.rename(columns={"value": "mic_um", "bacterium": "organism"})
    return df[["sequence", "mic_um", "organism"]].assign(source="grampa")


def load_dbaasp(path: str = "data/raw/dbaasp.csv") -> pd.DataFrame:
    """
    Returns columns: sequence, mic_um, organism, source='dbaasp'.
    See data/fetch_dbaasp.py for how to generate this file.
    """
    df = pd.read_csv(path)
    df = df.rename(columns={"SEQUENCE": "sequence", "MIC_VALUE": "mic_um", "TARGET": "organism"})
    return df[["sequence", "mic_um", "organism"]].assign(source="dbaasp")


def build_combined_mic_dataset(organism_filter: str = None, gram_filter: str = None,
                                organism_match_mode: str = "contains") -> pd.DataFrame:
    """
    Combines GRAMPA + DBAASP (DBAASP optional — skipped if not present yet),
    collapses multi-strain entries into a single representative MIC per
    peptide (median across organisms/sources), and log-transforms MIC for
    downstream binning.

    organism_filter: organism name to filter by, e.g. "E. coli" or "coli".
        With organism_match_mode="contains" (the default), this matches any
        organism string containing the given text, case-insensitively.
        Use organism_match_mode="exact" for a precise match instead.
    gram_filter: "positive" or "negative" — keeps only organisms of that
        Gram type (broader than organism_filter, still avoids Gram-mixing).
    Only one of organism_filter / gram_filter should be set.
    """
    frames = [load_grampa()]

    try:
        frames.append(load_dbaasp())
    except FileNotFoundError:
        print("Note: data/raw/dbaasp.csv not found — training on GRAMPA only for now.")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=["sequence", "mic_um"])
    combined = combined[combined["mic_um"] > 0]

    if organism_filter is not None:
        target = organism_filter.strip().lower()
        organism_lower = combined["organism"].str.strip().str.lower()
        if organism_match_mode == "contains":
            combined = combined[organism_lower.str.contains(target, na=False, regex=False)]
        elif organism_match_mode == "exact":
            combined = combined[organism_lower == target]
        else:
            raise ValueError(f"organism_match_mode must be 'contains' or 'exact', got {organism_match_mode!r}")
    elif gram_filter is not None:
        combined = combined[combined["organism"].apply(gram_of) == gram_filter]

    if len(combined) == 0:
        raise ValueError(
            f"No rows left after filtering (organism_filter={organism_filter!r}, "
            f"gram_filter={gram_filter!r}). Check the organism names actually "
            f"present in your data (df['organism'].unique())."
        )

    combined["log_mic"] = np.log10(combined["mic_um"])

    collapsed = (
        combined.groupby("sequence")["log_mic"]
        .median()
        .reset_index()
    )
    return collapsed


def compute_potency_cutoffs(df: pd.DataFrame):
    """
    Computes tercile log_mic cutoffs from a DataFrame. Call this on the
    TRAINING partition only, then pass the result to apply_potency_labels()
    for both train and test — computing cutoffs from the full dataset
    (including the test partition) before splitting is a data leakage bug:
    it lets test-set label boundaries be informed by test-set values.
    """
    low_cut, high_cut = df["log_mic"].quantile([1 / 3, 2 / 3])
    return float(low_cut), float(high_cut)


def apply_potency_labels(df: pd.DataFrame, low_cut: float, high_cut: float) -> pd.DataFrame:
    """Labels Low/Medium/High using cutoffs computed elsewhere (see compute_potency_cutoffs)."""
    def label(log_mic):
        if log_mic <= low_cut:      # low log-MIC = more potent
            return "High"
        elif log_mic <= high_cut:
            return "Medium"
        return "Low"

    df = df.copy()
    df["potency_class"] = df["log_mic"].apply(label)
    return df


def assign_potency_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience wrapper that computes cutoffs AND applies them to the same
    DataFrame. Fine for exploratory analysis on a single dataset, but NOT
    for train/test pipelines — there, compute cutoffs on the training
    partition only (compute_potency_cutoffs) and apply the same cutoffs to
    both partitions (apply_potency_labels), so test labels aren't defined
    using test data. See train/train_potency_model.py for the correct usage.
    """
    low_cut, high_cut = compute_potency_cutoffs(df)
    return apply_potency_labels(df, low_cut, high_cut)