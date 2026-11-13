"""Exact item-item Jaccard neighbours using a CSR inverted index and np.bincount."""
import numpy as np
from scipy import sparse


def exact_neighbors(item_user, user_item, query_item, top_n=50):
    """Return (item_ids, similarities), sorted by descending Jaccard then item ID.

    Both matrices must be binary CSR and transposes of one another.
    item_user shape: items x users; user_item shape: users x items.
    Only items sharing at least one positive user with query are enumerated.
    """
    if not sparse.isspmatrix_csr(item_user) or not sparse.isspmatrix_csr(user_item):
        raise TypeError('Both matrices must be scipy CSR')
    n_items, n_users = item_user.shape
    if user_item.shape != (n_users, n_items):
        raise ValueError('Matrix shapes do not match')
    if not 0 <= query_item < n_items:
        raise IndexError('query_item out of bounds')
    if top_n < 0:
        raise ValueError('top_n must be nonnegative')
    start, end = item_user.indptr[query_item:query_item+2]
    users = item_user.indices[start:end]
    if not len(users) or not top_n:
        return np.array([], dtype=np.int32), np.array([], dtype=np.float32)
    # Inverted index: expand only users who liked the query item.
    counts = np.bincount(
        np.concatenate([user_item.indices[user_item.indptr[u]:user_item.indptr[u+1]] for u in users]),
        minlength=n_items,
    )
    counts[query_item] = 0
    candidates = np.flatnonzero(counts)
    sizes = np.diff(item_user.indptr)
    union = sizes[query_item] + sizes[candidates] - counts[candidates]
    sims = counts[candidates] / union
    order = np.lexsort((candidates, -sims))[:top_n]
    return candidates[order].astype(np.int32), sims[order].astype(np.float32)


def exact_neighbors_for_queries(item_user, user_item, query_ids, top_n=50):
    """Fixed-width output for interoperable NPZ files; -1 pads missing IDs."""
    query_ids = np.asarray(query_ids, dtype=np.int32)
    ids = np.full((len(query_ids), top_n), -1, dtype=np.int32)
    scores = np.zeros((len(query_ids), top_n), dtype=np.float32)
    for row, q in enumerate(query_ids):
        n, s = exact_neighbors(item_user, user_item, int(q), top_n)
        ids[row, :len(n)], scores[row, :len(s)] = n, s
    return {'query_item_ids': query_ids, 'neighbour_ids': ids, 'neighbour_sims': scores}
