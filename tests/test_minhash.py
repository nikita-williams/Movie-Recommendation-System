"""MinHash checks (Member 3): estimates must track true Jaccard, and agree with datasketch's accuracy."""
import numpy as np
import pytest
from scipy import sparse

from src.minhash import estimate_jaccard, exact_jaccard_pairs, minhash_signatures


def planted_pairs(n_pairs=100, size=150, n_users=20000, seed=0):
    """Item pairs with true Jaccard spread over (0, 1): rows 2p and 2p+1 form pair p."""
    rng = np.random.default_rng(seed)
    rows, cols = [], []
    for p in range(n_pairs):
        shared = int(rng.integers(5, size))  # overlap anywhere from tiny to nearly identical
        users = rng.choice(n_users, 2 * size - shared, replace=False)
        a, b = users[:size], np.concatenate([users[:shared], users[size:]])
        for row, s in ((2 * p, a), (2 * p + 1, b)):
            rows += [row] * len(s)
            cols += list(s)
    m = sparse.csr_matrix((np.ones(len(rows), dtype=np.uint8), (rows, cols)), shape=(2 * n_pairs, n_users))
    return m, np.arange(0, 2 * n_pairs, 2), np.arange(1, 2 * n_pairs, 2)


def test_signature_shape_and_determinism():
    m, _, _ = planted_pairs(10)
    s1, s2 = minhash_signatures(m, k=128, seed=3), minhash_signatures(m, k=128, seed=3)
    assert s1.shape == (20, 128) and s1.dtype == np.uint32
    assert np.array_equal(s1, s2)
    assert not np.array_equal(s1, minhash_signatures(m, k=128, seed=4))


def test_identical_items_match_and_disjoint_items_do_not():
    m = sparse.csr_matrix(np.array([[1, 1, 1, 0, 0, 0],
                                    [1, 1, 1, 0, 0, 0],
                                    [0, 0, 0, 1, 1, 1]], dtype=np.uint8))
    sig = minhash_signatures(m, k=128)
    assert estimate_jaccard(sig, 0, 1) == 1.0
    assert estimate_jaccard(sig, 0, 2) < 0.05


def test_empty_item_does_not_corrupt_neighbours():
    m = sparse.csr_matrix(np.array([[1, 1, 0], [0, 0, 0], [0, 1, 1]], dtype=np.uint8))
    sig = minhash_signatures(m, k=16)
    assert (sig[1] == np.uint32(0xFFFFFFFF)).all()
    assert (sig[0] != np.uint32(0xFFFFFFFF)).all() and (sig[2] != np.uint32(0xFFFFFFFF)).all()


def test_estimate_matches_true_jaccard_on_100_random_pairs():
    k = 128
    m, i, j = planted_pairs(100)
    true = exact_jaccard_pairs(m, i, j)
    est = estimate_jaccard(minhash_signatures(m, k=k, seed=1), i, j)
    rmse = np.sqrt(((est - true) ** 2).mean())
    expected = np.sqrt((true * (1 - true) / k).mean())      # theoretical standard error of the estimator
    assert abs((est - true).mean()) < 0.015, 'estimator should be unbiased'
    assert rmse < 1.25 * expected, f'rmse {rmse:.4f} vs theory {expected:.4f}'
    assert np.mean(np.abs(est - true) <= 0.1) >= 0.9


def test_matches_datasketch_accuracy():
    datasketch = pytest.importorskip('datasketch')
    k = 128
    m, i, j = planted_pairs(60, seed=5)
    true = exact_jaccard_pairs(m, i, j)
    ours = estimate_jaccard(minhash_signatures(m, k=k, seed=2), i, j)

    def sketch(x):
        h = datasketch.MinHash(num_perm=k, seed=1)
        for u in m[x].indices:
            h.update(int(u).to_bytes(4, 'little'))
        return h
    theirs = np.array([sketch(a).jaccard(sketch(b)) for a, b in zip(i, j)])
    rmse_ours, rmse_theirs = (np.sqrt(((e - true) ** 2).mean()) for e in (ours, theirs))
    assert rmse_ours < 0.06 and rmse_theirs < 0.06
    assert abs(rmse_ours - rmse_theirs) < 0.02, 'same accuracy as the reference library'
