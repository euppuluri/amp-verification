"""
Homology-aware train/test splitting.

Prevents near-identical sequences from ending up on both sides of a
train/test split (which would inflate reported accuracy).

If a real `cd-hit` binary is installed and on PATH, this uses it directly
(the standard, rigorous tool used in AMP literature). If not, it falls
back to a dependency-free k-mer Jaccard similarity approximation so the
pipeline still works without asking you to install and compile a separate
binary first.

To install real CD-HIT:
    conda install -c bioconda cd-hit
    (or build from https://github.com/weizhongli/cdhit)
"""

import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def _cdhit_available() -> bool:
    return shutil.which("cd-hit") is not None


def _cluster_with_cdhit(sequences: List[str], similarity_threshold: float = 0.5) -> Dict[str, int]:
    """
    Runs real CD-HIT and parses its .clstr output into {sequence: cluster_id}.

    CD-HIT's -c (identity threshold) requires a matching -n (word length):
    -c 0.7-1.0 -> -n 5, -c 0.6-0.7 -> -n 4, -c 0.5-0.6 -> -n 3, -c 0.4-0.5 -> -n 2
    """
    if similarity_threshold >= 0.7:
        word_length = 5
    elif similarity_threshold >= 0.6:
        word_length = 4
    elif similarity_threshold >= 0.5:
        word_length = 3
    else:
        word_length = 2

    with tempfile.TemporaryDirectory() as tmp:
        input_fasta = Path(tmp) / "input.fasta"
        output_prefix = Path(tmp) / "clustered"

        with open(input_fasta, "w") as f:
            for i, seq in enumerate(sequences):
                f.write(f">seq{i}\n{seq}\n")

        subprocess.run(
            [
                "cd-hit", "-i", str(input_fasta), "-o", str(output_prefix),
                "-c", str(similarity_threshold), "-n", str(word_length),
            ],
            check=True,
            capture_output=True,
        )

        clstr_path = Path(f"{output_prefix}.clstr")
        cluster_of: Dict[str, int] = {}
        current_cluster = -1

        with open(clstr_path) as f:
            for line in f:
                if line.startswith(">Cluster"):
                    current_cluster += 1
                else:
                    # e.g. "0	37aa, >seq12... *"  -- extract the seqN index
                    seq_index = int(line.split(">seq")[1].split("...")[0])
                    cluster_of[sequences[seq_index]] = current_cluster

        return cluster_of


def _kmer_set(sequence: str, k: int = 4) -> set:
    if len(sequence) < k:
        return {sequence}
    return {sequence[i:i + k] for i in range(len(sequence) - k + 1)}


def _jaccard_similarity(set_a: set, set_b: set) -> float:
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union


def _cluster_with_kmer_fallback(sequences: List[str], similarity_threshold: float = 0.5, k: int = 4) -> Dict[str, int]:
    """
    Greedy single-linkage clustering by k-mer Jaccard similarity — used
    only when a real cd-hit binary isn't available.
    """
    kmer_sets = {seq: _kmer_set(seq, k) for seq in sequences}
    cluster_representatives: List[str] = []
    cluster_of: Dict[str, int] = {}

    for seq in sequences:
        assigned = False
        for cluster_id, rep_seq in enumerate(cluster_representatives):
            if _jaccard_similarity(kmer_sets[seq], kmer_sets[rep_seq]) >= similarity_threshold:
                cluster_of[seq] = cluster_id
                assigned = True
                break
        if not assigned:
            cluster_of[seq] = len(cluster_representatives)
            cluster_representatives.append(seq)

    return cluster_of


def cluster_sequences(sequences: List[str], similarity_threshold: float = 0.5, k: int = 4) -> Dict[str, int]:
    """
    Returns {sequence: cluster_id}. Uses real CD-HIT if installed, else
    falls back to a k-mer Jaccard approximation.
    """
    unique_sequences = list(dict.fromkeys(sequences))  # dedupe, preserve order

    if _cdhit_available():
        try:
            return _cluster_with_cdhit(unique_sequences, similarity_threshold)
        except Exception as e:
            print(f"cd-hit available but failed ({e}); falling back to k-mer clustering.")

    return _cluster_with_kmer_fallback(unique_sequences, similarity_threshold, k)


def homology_partition_split(df, sequence_col: str = "sequence", test_frac: float = 0.2,
                              similarity_threshold: float = 0.5, random_state: int = 42):
    """
    Splits a DataFrame into train/test such that no two sequences from the
    same similarity cluster appear on both sides.
    """
    import random

    sequences = df[sequence_col].tolist()
    cluster_of = cluster_sequences(sequences, similarity_threshold=similarity_threshold)

    clusters_to_rows = defaultdict(list)
    for idx, seq in zip(df.index, df[sequence_col]):
        clusters_to_rows[cluster_of[seq]].append(idx)

    cluster_ids = list(clusters_to_rows.keys())
    random.Random(random_state).shuffle(cluster_ids)

    target_test_size = int(len(df) * test_frac)
    test_indices = []
    for cluster_id in cluster_ids:
        if len(test_indices) >= target_test_size:
            break
        test_indices.extend(clusters_to_rows[cluster_id])

    test_df = df.loc[test_indices]
    train_df = df.drop(test_indices)
    return train_df, test_df