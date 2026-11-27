"""LSH checks (Member 3): measured candidate rate follows the S-curve; re-scoring matches exact Jaccard."""
import numpy as np
from scipy import sparse

from src.exact_jaccard import exact_neighbors
from src.lsh import SETTINGS, LSHIndex, theoretical_probability, threshold
from src.minhash import exact_jaccard_pairs, minhash_signatures
from src.rescore import Rescorer, lsh_neighbors_for_queries
from tests.test_minhash import planted_pairs


def test_settings_and_thresholds():
    assert SETTINGS == [(64, 2), (32, 4), (16, 8), (8, 16)]
    assert all(b * r == 128 for b, r in SETTINGS)
    assert np.allclose([threshold(b, r) for b, r in SETTINGS], [0.125, 0.4204, 0.7071, 0.8781], atol=1e-3)


def test_measured_candidate_rate_follows_s_curve():
    m, i, j = planted_pairs(400, size=150, seed=11)
    true = exact_jaccard_pairs(m, i, j)
    sig = minhash_signatures(m, k=128, seed=0)
    for b, r in SETTINGS:
        index = LSHIndex(sig, b, r)
        measured = index.is_candidate(i, j)
        theory = theoretical_probability(true, b, r)
        assert abs(measured.mean() - theory.mean()) < 0.06, (b, r)
        # also check the shape: pairs well above the threshold nearly always collide, well below rarely do
        hi, lo = true > threshold(b, r) + 0.25, true < threshold(b, r) - 0.25
        if hi.sum() >= 10:
            assert measured[hi].mean() > 0.9
        if lo.sum() >= 10:
            assert measured[lo].mean() < 0.1


def test_bigger_r_gives_fewer_candidates():
    m, _, _ = planted_pairs(100, seed=3)
    sig = minhash_signatures(m, k=128, seed=0)
    sizes = []
    for b, r in SETTINGS:
        index = LSHIndex(sig, b, r)
        sizes.append(np.mean([len(index.candidates(q)) for q in range(0, 200, 4)]))
    assert sizes == sorted(sizes, reverse=True), sizes


def test_candidates_exclude_self_and_agree_with_pairwise_check():
    m, _, _ = planted_pairs(50, seed=4)
    index = LSHIndex(minhash_signatures(m, k=128, seed=0), 32, 4)
    for q in (0, 7, 31):
        c = index.candidates(q)
        assert q not in c
        others = np.arange(m.shape[0])
        others = others[others != q]
        assert set(others[index.is_candidate(np.full(len(others), q), others)]) == set(c.tolist())


def test_rescoring_all_items_equals_exact_neighbors():
    rng = np.random.default_rng(0)
    item_user = sparse.csr_matrix((rng.random((60, 400)) < 0.12).astype(np.uint8))
    item_user = item_user[np.diff(item_user.indptr) > 0].tocsr()
    user_item = item_user.T.tocsr()
    rescorer = Rescorer(item_user)
    for q in (0, 5, 20):
        everything = np.setdiff1d(np.arange(item_user.shape[0]), [q]).astype(np.int32)
        ids, sims = rescorer.neighbors(q, everything, top_n=50)
        e_ids, e_sims = exact_neighbors(item_user, user_item, q, top_n=50)
        assert np.array_equal(ids, e_ids) and np.allclose(sims, e_sims)


def test_lsh_neighbours_are_true_jaccard_values_and_padded_like_exact():
    m, _, _ = planted_pairs(40, seed=8)
    index = LSHIndex(minhash_signatures(m, k=128, seed=0), 32, 4)
    out = lsh_neighbors_for_queries(m, index, [0, 1, 2], top_n=50)
    assert out['neighbour_ids'].shape == (3, 50) and out['neighbour_sims'].dtype == np.float32
    for row, q in enumerate(out['query_item_ids']):
        ids, sims = out['neighbour_ids'][row], out['neighbour_sims'][row]
        valid = ids >= 0
        assert np.allclose(sims[valid], exact_jaccard_pairs(m, np.full(valid.sum(), q), ids[valid]), atol=1e-6)
        assert (sims[~valid] == 0).all()
