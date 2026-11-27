"""Exact Jaccard on the LSH candidate set only (Member 3).

Output format is identical to Member 2's exact_neighbors / exact_neighbors_for_queries, so with
a candidate set that contains every true neighbour the result equals the exact result.
"""
import numpy as np
from scipy import sparse

from src.candidates import candidate_sets


class Rescorer:
    """Exact Jaccard between one query item and a list of candidate items."""

    def __init__(self, item_user):
        if not sparse.isspmatrix_csr(item_user):
            raise TypeError('item_user must be scipy CSR')
        self.item_user = item_user
        self.sizes = np.diff(item_user.indptr)
        self._mask = np.zeros(item_user.shape[1], dtype=np.int32)

    def neighbors(self, query, cands, top_n=50):
        """(item_ids, similarities) of the top_n candidates, sorted by descending Jaccard then item id."""
        empty = np.array([], dtype=np.int32), np.array([], dtype=np.float32)
        m = self.item_user
        users = m.indices[m.indptr[query]:m.indptr[query + 1]]
        if len(cands) == 0 or len(users) == 0 or top_n == 0:
            return empty
        self._mask[users] = 1
        counts = m[cands].dot(self._mask)            # |query users & candidate users| for every candidate
        self._mask[users] = 0
        union = self.sizes[query] + self.sizes[cands] - counts
        keep = counts > 0
        cands, counts, union = cands[keep], counts[keep], union[keep]
        sims = counts / union
        order = np.lexsort((cands, -sims))[:top_n]   # same tie-break as exact_jaccard.py
        return cands[order].astype(np.int32), sims[order].astype(np.float32)


def lsh_neighbors_for_queries(item_user, index, query_ids, top_n=50):
    """Fixed-width output like exact_neighbors_for_queries ('-1' pads missing ids), plus candidate sizes."""
    query_ids = np.asarray(query_ids, dtype=np.int32)
    rescorer = Rescorer(item_user)
    ids = np.full((len(query_ids), top_n), -1, dtype=np.int32)
    scores = np.zeros((len(query_ids), top_n), dtype=np.float32)
    cands = candidate_sets(index, query_ids)
    for row, q in enumerate(query_ids):
        n, s = rescorer.neighbors(int(q), cands[row], top_n)
        ids[row, :len(n)], scores[row, :len(s)] = n, s
    return {'query_item_ids': query_ids, 'neighbour_ids': ids, 'neighbour_sims': scores,
            'candidate_sizes': np.array([len(c) for c in cands], dtype=np.int64)}
