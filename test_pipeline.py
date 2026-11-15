"""Quick checks on fake MovieLens-like data (no download needed):  python test_pipeline.py"""
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
import scipy.sparse as sp

from minhash import estimate_jaccard, exact_jaccard, minhash_signatures


def fake_ratings(n_users=4000, n_movies=1500, seed=1):
    rng = np.random.default_rng(seed)
    pop = np.minimum(rng.zipf(1.6, n_movies), 200).astype(float)   # a few very popular movies
    pop /= pop.sum()
    parts = []
    for u in range(1, n_users + 1):
        n = min(int(20 + rng.pareto(1.5) * 15), n_movies // 2)
        parts.append(pd.DataFrame({
            "userId": u, "movieId": rng.choice(n_movies, n, replace=False, p=pop) + 1,
            "rating": rng.choice(np.arange(0.5, 5.01, 0.5), n),
            "timestamp": np.sort(rng.integers(8e8, 1.57e9, n))}))
    return pd.concat(parts)


def test_preprocess(tmp):
    path = f"{tmp}/ratings.csv"
    fake_ratings().to_csv(path, index=False)
    subprocess.run([sys.executable, "preprocess.py", "--ratings", path, "--out", f"{tmp}/out"],
                   check=True, stdout=subprocess.DEVNULL)
    load = lambda tag: (sp.load_npz(f"{tmp}/out/item_user_{tag}.npz"),
                        np.load(f"{tmp}/out/items_{tag}.npy"))
    (m10, i10), (m30, i30), (m100, i100) = load("10"), load("30"), load("100")
    for m in (m10, m30, m100):
        assert np.diff(m.indptr).min() >= 10, "every movie needs >= 10 likes"
    assert set(i10) <= set(i30) <= set(i100), "bigger groups keep the smaller group's movies"
    assert m10.nnz < m30.nnz < m100.nnz, "scales should grow"
    eval_users = np.load(f"{tmp}/out/eval_users.npy")
    test = pd.read_csv(f"{tmp}/out/eval_test.csv")
    assert set(test["user_idx"]) == set(eval_users.tolist()), "every eval user has test data"
    print("preprocess OK")
    return m100


def test_minhash(mat, k=128):
    rng = np.random.default_rng(0)
    co = (mat @ mat.T).tocsr()                       # pairs of movies that share users
    i = rng.integers(0, mat.shape[0], 2000)
    j = np.array([rng.choice(co[x].indices) for x in i])
    ok = i != j
    i, j = i[ok], j[ok]
    true = exact_jaccard(mat, i, j)
    est = estimate_jaccard(minhash_signatures(mat, k=k), i, j)
    rmse = np.sqrt(((est - true) ** 2).mean())
    print(f"MinHash typical error {rmse:.3f}  (theory about {np.sqrt((true*(1-true)/k).mean()):.3f})")
    assert abs((est - true).mean()) < 0.01, "MinHash should be unbiased"
    try:  # optional cross-check against the datasketch library
        from datasketch import MinHash
        def sk(x):
            h = MinHash(num_perm=k, seed=1)
            for u in mat[x].indices:
                h.update(int(u).to_bytes(4, "little"))
            return h
        d = np.array([sk(a).jaccard(sk(b)) for a, b in zip(i[:200], j[:200])])
        print(f"datasketch error {np.sqrt(((d - true[:200]) ** 2).mean()):.3f} (should be similar)")
    except ImportError:
        print("datasketch not installed, skipped cross-check")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        matrix = test_preprocess(tmp)
    test_minhash(matrix)
    print("all checks passed")
