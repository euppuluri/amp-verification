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

DBAASP has no bulk CSV endpoint — export search results manually from
    https://dbaasp.org/
and save as data/raw/dbaasp.csv. DBAASP is OPTIONAL: if that file doesn't
exist yet, this loader just trains on GRAMPA alone rather than erroring.
"""

import pandas as pd
import numpy as np


def load_grampa(path: str = "data/raw/grampa.csv") -> pd.DataFrame:
    """Returns columns: sequence, mic_um, organism, source='grampa'."""
    df = pd.read_csv(path)
    df = df.rename(columns={"value": "mic_um", "bacterium": "organism"})
    return df[["sequence", "mic_um", "organism"]].assign(source="grampa")


def load_dbaasp(path: str = "data/raw/dbaasp.csv") -> pd.DataFrame:
    """
    Returns columns: sequence, mic_um, organism, source='dbaasp'.

    TODO: DBAASP's export column names depend on your search/export
    settings at https://dbaasp.org/ — inspect the downloaded CSV's header
    row and adjust the `rename` mapping below to match.
    """
    df = pd.read_csv(path)
    df = df.rename(columns={"SEQUENCE": "sequence", "MIC_VALUE": "mic_um", "TARGET": "organism"})
    return df[["sequence", "mic_um", "organism"]].assign(source="dbaasp")


def build_combined_mic_dataset() -> pd.DataFrame:
    """
    Combines GRAMPA + DBAASP (DBAASP optional — skipped if not present yet),
    collapses multi-strain entries into a single representative MIC per
    peptide (median across organisms/sources), and log-transforms MIC for
    downstream binning.
    """
    frames = [load_grampa()]

    try:
        frames.append(load_dbaasp())
    except FileNotFoundError:
        print("Note: data/raw/dbaasp.csv not found — training on GRAMPA only for now.")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=["sequence", "mic_um"])
    combined = combined[combined["mic_um"] > 0]

    combined["log_mic"] = np.log10(combined["mic_um"])

    collapsed = (
        combined.groupby("sequence")["log_mic"]
        .median()
        .reset_index()
    )
    return collapsed


def assign_potency_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bins log_mic into Low/Medium/High potency classes using tercile cutoffs
    computed on the training distribution.
    """
    low_cut, high_cut = df["log_mic"].quantile([1 / 3, 2 / 3])

    def label(log_mic):
        if log_mic <= low_cut:      # low log-MIC = more potent
            return "High"
        elif log_mic <= high_cut:
            return "Medium"
        return "Low"

    df["potency_class"] = df["log_mic"].apply(label)
    return df
