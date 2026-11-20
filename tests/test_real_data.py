
import time
import numpy as np
from scipy import sparse

from src.exact_jaccard import exact_neighbors


def test_exact_jaccard(scale=10):
    path = f"data/processed/item_user_{scale}.npz"

    item_user = sparse.load_npz(path).tocsr()
    user_item = item_user.T.tocsr()
    movie_ids = np.load(f"data/processed/items_{scale}.npy")

    # Test a popular movie
    popularity = np.diff(item_user.indptr)
    query = int(np.argmax(popularity))

    start = time.perf_counter()
    neighbors, scores = exact_neighbors(
        item_user, user_item, query, top_n=50
    )
    elapsed = time.perf_counter() - start

    # Verify each similarity independently
    query_users = set(item_user[query].indices)

    for idx, score in zip(neighbors, scores):
        other_users = set(item_user[int(idx)].indices)
        intersection = len(query_users & other_users)
        union = len(query_users | other_users)
        expected = intersection / union if union else 0

        assert np.isclose(score, expected, atol=1e-7)

    # Verify descending similarity
    assert np.all(scores[:-1] >= scores[1:] - 1e-7)

    # Verify no duplicate neighbours or self-match
    assert len(neighbors) == len(set(neighbors))
    assert query not in neighbors

    print(f"\nDataset: {scale}%")
    print(f"Query movie ID: {movie_ids[query]}")
    print(f"Neighbours found: {len(neighbors)}")
    print(f"Runtime: {elapsed:.4f} seconds")
    print("PASS: Exact Jaccard similarity checks")


if __name__ == "__main__":
    for scale in [10, 30, 100]:
        test_exact_jaccard(scale)
