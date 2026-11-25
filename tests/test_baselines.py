import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
from scipy.sparse import csr_matrix
from exact_jaccard import exact_neighbors, exact_neighbors_for_queries
from popularity import popular_recommendations
from scoring import score_recommendations, lookup_from_arrays
from timing import measure

# Users: u0 likes 0,1; u1 likes 0,1,2; u2 likes 1,2; u3 likes 2,3
X = csr_matrix(np.array([[1,1,0,0],[1,1,1,0],[0,1,1,0],[0,0,1,1]], dtype=np.uint8))
Y = X.T.tocsr()

def test_jaccard():
    ids, sims = exact_neighbors(Y, X, 0)
    assert ids.tolist() == [1,2]
    np.testing.assert_allclose(sims, [2/3, 1/4])
    assert 0 not in ids

def test_popularity():
    ids, scores = popular_recommendations(X, 0, 10)
    assert ids.tolist() == [2,3]
    assert scores.tolist() == [3,1]

def test_scoring():
    arrays = exact_neighbors_for_queries(Y, X, [0,1], 3)
    lookup = lookup_from_arrays(**arrays)
    ids, scores = score_recommendations([0,1], [0,1], lookup, 10)
    assert ids.tolist() == [2]
    np.testing.assert_allclose(scores, [0.25 + 2/4], atol=1e-6)

def test_timing():
    records = []
    with measure(records, 'example'):
        sum(range(100))
    assert records[0]['elapsed_s'] >= 0
