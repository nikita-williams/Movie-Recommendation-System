"""Popularity baseline: rank unseen items by positive training-interaction count."""
import numpy as np
from scipy import sparse


def popular_recommendations(user_item, user_id, k=10):
    if not sparse.isspmatrix_csr(user_item):
        raise TypeError('user_item must be CSR')
    if not 0 <= user_id < user_item.shape[0]:
        raise IndexError('user_id out of bounds')
    if k < 0:
        raise ValueError('k must be nonnegative')
    counts = np.asarray(user_item.getnnz(axis=0)).ravel()
    seen = user_item.indices[user_item.indptr[user_id]:user_item.indptr[user_id+1]]
    counts = counts.astype(np.int64, copy=True)
    counts[seen] = -1
    candidates = np.flatnonzero(counts > 0)
    order = np.lexsort((candidates, -counts[candidates]))[:k]
    selected = candidates[order].astype(np.int32)
    return selected, counts[selected].astype(np.float32)
