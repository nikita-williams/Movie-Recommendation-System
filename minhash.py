"""MinHash, written from scratch (used by the LSH step).

Each movie is the set of users who liked it. Jaccard(A, B) = |A and B| / |A or B|.
Instead of comparing big sets, each movie is shrunk to k numbers (its signature):
for each of k random hash functions, hash every user who liked the movie and keep
the smallest value. For two movies, P(the minimum matches) = their Jaccard similarity,
so the fraction of matching positions in two signatures estimates it.
"""
import numpy as np
import scipy.sparse as sp

P = 2**31 - 1  # a prime; hash is h(u) = (a*u + b) mod P (products stay below 2^62, so uint64 is safe)


def minhash_signatures(item_user, k=128, seed=0):
    """item_user: movies x users sparse table (every movie needs at least one user).
    Returns a (k, n_movies) uint32 array."""
    m = sp.csr_matrix(item_user)  # row i's users are m.indices[m.indptr[i]:m.indptr[i+1]]
    rng = np.random.default_rng(seed)
    a = rng.integers(1, P, size=k, dtype=np.uint64)
    b = rng.integers(0, P, size=k, dtype=np.uint64)
    users = np.arange(m.shape[1], dtype=np.uint64)
    sig = np.empty((k, m.shape[0]), dtype=np.uint32)
    for j in range(k):
        h = ((a[j] * users + b[j]) % np.uint64(P)).astype(np.uint32)  # hash every user once
        sig[j] = np.minimum.reduceat(h[m.indices], m.indptr[:-1])     # smallest hash per movie
    return sig


def estimate_jaccard(sig, i, j):
    """Estimated Jaccard of movies i and j (ints or index arrays): share of matching rows."""
    return (sig[:, i] == sig[:, j]).mean(axis=0)


def exact_jaccard(item_user, i, j):
    """True Jaccard, used to check MinHash on a sample: c / (|A| + |B| - c),
    where c is the number of users who liked both movies."""
    m = sp.csr_matrix(item_user, dtype=np.int32)
    size = np.diff(m.indptr)
    i, j = np.atleast_1d(i), np.atleast_1d(j)
    c = np.asarray(m[i].multiply(m[j]).sum(axis=1)).ravel()
    return c / (size[i] + size[j] - c)
