"""MinHash signatures, written from scratch (Member 3).

Each item (movie) is the set of users who positively rated it. Jaccard(A, B) = |A & B| / |A | B|.
Every item is shrunk to k numbers: for each of k random hash functions, hash every user in the
item's set and keep the smallest value. For two items, P(minimums match) = Jaccard(A, B), so the
share of matching positions in two signatures is an unbiased estimate of the Jaccard similarity.

Hash family: h(u) = (a*u + b) mod P with P = 2**31 - 1 (prime), a in [1, P), b in [0, P).
a*u stays below 2**62 for user indices below 2**31, so uint64 arithmetic never overflows.
Signatures are stored as (n_items, k) uint32, one row per item.
"""
import numpy as np
from scipy import sparse

P = 2**31 - 1
EMPTY = np.uint32(0xFFFFFFFF)  # signature value of an item with no users (never equals a real hash, which is < 2**31)


def hash_params(k=128, seed=0):
    """Random (a, b) for k hash functions; the same seed always gives the same functions."""
    rng = np.random.default_rng(seed)
    a = rng.integers(1, P, size=k, dtype=np.uint64)
    b = rng.integers(0, P, size=k, dtype=np.uint64)
    return a, b


def minhash_signatures(item_user, k=128, seed=0):
    """Signature matrix for a binary items x users CSR matrix. Returns (n_items, k) uint32."""
    m = sparse.csr_matrix(item_user)
    n_items, n_users = m.shape
    if n_users >= 2**31:
        raise ValueError('user index must be below 2**31')
    a, b = hash_params(k, seed)
    users = np.arange(n_users, dtype=np.uint64)
    sizes = np.diff(m.indptr)
    nonempty = np.flatnonzero(sizes > 0)
    sig = np.full((n_items, k), EMPTY, dtype=np.uint32)
    if len(nonempty) == 0:
        return sig
    # reduceat needs the start offset of each non-empty row; empty rows are skipped so they cannot
    # pick up a neighbouring row's value.
    starts = m.indptr[nonempty]
    for j in range(k):
        h = ((a[j] * users + b[j]) % np.uint64(P)).astype(np.uint32)  # hash every user once
        sig[nonempty, j] = np.minimum.reduceat(h[m.indices], starts)    # smallest hash per item
    return sig


def estimate_jaccard(sig, i, j):
    """Estimated Jaccard of items i and j (ints or index arrays): share of matching signature positions."""
    return (sig[i] == sig[j]).mean(axis=-1)


def exact_jaccard_pairs(item_user, i, j):
    """True Jaccard for index arrays i, j (used only to check MinHash, never in the pipeline)."""
    m = sparse.csr_matrix(item_user, dtype=np.int32)
    size = np.diff(m.indptr)
    i, j = np.atleast_1d(i), np.atleast_1d(j)
    c = np.asarray(m[i].multiply(m[j]).sum(axis=1)).ravel()
    union = size[i] + size[j] - c
    return np.where(union > 0, c / np.maximum(union, 1), 0.0)
