"""Candidate lookup and candidate statistics for LSH (Member 3)."""
import numpy as np


def candidate_sets(index, query_ids):
    """Candidate item rows for each query item (one array per query, query itself excluded)."""
    return [index.candidates(int(q)) for q in query_ids]


def candidate_sizes(cands):
    return np.array([len(c) for c in cands], dtype=np.int64)


def candidate_recall(cand, true_ids):
    """|candidates & true top-n| / |true top-n|.

    true_ids is Member 2's exact top-n for the same query ('-1' padding is ignored). The divisor is the
    number of true neighbours that exist (equal to 50 whenever the item has 50 neighbours), so items with
    fewer than 50 overlapping neighbours are not penalised. Returns nan if the item has no neighbours.
    """
    true_ids = np.asarray(true_ids)
    true_ids = true_ids[true_ids >= 0]
    if len(true_ids) == 0:
        return float('nan')
    return float(np.isin(true_ids, cand).sum() / len(true_ids))
