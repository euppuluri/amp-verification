"""
Loads and harmonizes MIC-labeled peptide data from GRAMPA and DBAASP.

GRAMPA download (no account needed):
    https://raw.githubusercontent.com/zswitten/Antimicrobial-Peptides/master/data/grampa.csv
Save it to: data/raw/grampa.csv

Real GRAMPA columns: bacterium, sequence, strain, value, database,
url_source, modifications, unit (always 'uM'), is_modified,
has_unusual_modification, has_cterminal_amidation, datasource_has_modifications

DBAASP has no bulk CSV endpoint — export manually from https://dbaasp.org/
and save as data/raw/dbaasp.csv. Check the actual column headers of your
export and adjust load_dbaasp's rename mapping below to match.
"""

import pandas as pd
import numpy as np


def load_grampa(path: str = "data/raw/grampa.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={"value": "mic_um", "bacterium": "organism"})
    return df[["sequence", "mic_um", "organism"]].assign(source="grampa")


def load_dbaasp(path: str = "data/raw/dbaasp.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={"SEQUENCE": "sequence", "MIC_VALUE": "mic_um", "TARGET": "organism"})
    return df[["sequence", "mic_um", "organism"]].assign(source="dbaasp")


def build_combined_mic_dataset() -> pd.DataFrame:
    combined = pd.concat([load_grampa(), load_dbaasp()], ignore_index=True)
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
    low_cut, high_cut = df["log_mic"].quantile([1 / 3, 2 / 3])

    def label(log_mic):
        if log_mic <= low_cut:
            return "High"
        elif log_mic <= high_cut:
            return "Medium"
        return "Low"

    df["potency_class"] = df["log_mic"].apply(label)
    return df