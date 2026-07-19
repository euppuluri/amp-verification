"""
Homology-aware train/test splitting.

Prevents near-identical sequences from ending up on both sides of a
train/test split (which would inflate reported accuracy). Uses k-mer
Jaccard similarity as a dependency-free approximation of CD-HIT.
"""

from collections import defaultdict


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


def cluster_sequences(sequences: list[str], similarity_threshold: float = 0.5, k: int = 4) -> dict[str, int]:
    kmer_sets = {seq: _kmer_set(seq, k) for seq in sequences}
    cluster_representatives: list[str] = []
    cluster_of: dict[str, int] = {}

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


def homology_partition_split(df, sequence_col: str = "sequence", test_frac: float = 0.2,
                              similarity_threshold: float = 0.5, random_state: int = 42):
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