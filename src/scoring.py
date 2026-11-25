"""One shared recommendation scoring rule for exact and LSH neighbours."""
import numpy as np


def score_recommendations(query_items, seen_items, neighbor_lookup, k=10):
    """Sum Jaccard similarities across query items; exclude previously liked items.

    neighbor_lookup maps query item -> (neighbor_ids, similarities).
    Caller supplies up to 20 most recent positive training items as query_items.
    """
    if k < 0:
        raise ValueError('k must be nonnegative')
    seen = set(map(int, seen_items))
    totals = {}
    for query in query_items:
        ids, sims = neighbor_lookup.get(int(query), ([], []))
        for item, similarity in zip(ids, sims):
            item = int(item)
            if item < 0 or item in seen or similarity <= 0:
                continue
            totals[item] = totals.get(item, 0.0) + float(similarity)
    ranked = sorted(totals, key=lambda item: (-totals[item], item))[:k]
    return np.asarray(ranked, dtype=np.int32), np.asarray([totals[i] for i in ranked], dtype=np.float32)


def lookup_from_arrays(query_item_ids, neighbour_ids, neighbour_sims):
    return {int(q): (neighbour_ids[i], neighbour_sims[i]) for i, q in enumerate(query_item_ids)}
