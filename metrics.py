
import numpy as np
import pandas as pd


""" metrics 

Relevance is binary: a recommended movie is a hit if it is in the user's held-out test positives.

    Precision-K = hits in top K / K(a short list is NOT rewarded: we still divide by K)
    Recall-K    = hits in top K / number of test positives
    NDCG-K      = DCG-K / IDCG-K,  DCG = sum over hits of 1 / log2(rank + 1)
                  IDCG = DCG of a perfect list with min(K, number of test positives) hits

Every evaluation user is scored. A user with no recommendations (this happens with strict LSH
settings) scores 0; dropping them would make a method that often returns nothing look better.
"""


KS = (10, 20)


def _hits(ranked, relevant, k):
    """0/1 array over the top k of a ranked list."""
    if k < 1:
        raise ValueError('k must be positive')
    top = list(ranked)[:k]
    if len(set(top)) != len(top):
        raise ValueError('duplicate item in a recommendation list')
    return np.array([item in relevant for item in top], dtype=float)


def precision_at_k(ranked, relevant, k):
    return float(_hits(ranked, relevant, k).sum() / k)


def recall_at_k(ranked, relevant, k):
    if not relevant:
        raise ValueError('recall is undefined for a user with no test items')
    return float(_hits(ranked, relevant, k).sum() / len(relevant))


def ndcg_at_k(ranked, relevant, k):
    if not relevant:
        raise ValueError('NDCG is undefined for a user with no test items')
    hits = _hits(ranked, relevant, k)
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = (hits * discounts[:len(hits)]).sum()
    idcg = discounts[:min(k, len(relevant))].sum()
    return float(dcg / idcg)


def candidate_recall(candidates, exact_neighbours):
    """Share of the exact top neighbours found among the LSH candidates ('-1' padding ignored).

    Same definition as src/candidates.py; kept here so it can be checked independently.
    Returns nan when the query item has no true neighbours.
    """
    truth = {int(i) for i in exact_neighbours if int(i) >= 0}
    if not truth:
        return float('nan')
    return len(truth & {int(i) for i in candidates}) / len(truth)


def evaluate_users(recs, test, eval_users, ks=KS):
    """Per-user metrics for ONE recommendation file.

    recs: DataFrame user_idx, movieId, rank 
    test: DataFrame user_idx, movieId (eval_test.csv)
    eval_users: the users to score (from eval_users.npy, never taken from the CSV)
    """
    users = [int(u) for u in eval_users]
    unknown = set(recs['user_idx'].astype(int)) - set(users)
    if unknown:
        raise ValueError(f'recommendations for users outside the evaluation sample: {sorted(unknown)[:5]}')
    if recs.duplicated(['user_idx', 'movieId']).any() or recs.duplicated(['user_idx', 'rank']).any():
        raise ValueError('duplicate movie or rank for a user')
    relevant = {int(u): set(map(int, g)) for u, g in test.groupby('user_idx')['movieId']}
    ordered = recs.sort_values(['user_idx', 'rank'])
    ranked = {int(u): [int(m) for m in g] for u, g in ordered.groupby('user_idx')['movieId']}
    rows = []
    for u in users:
        if u not in relevant:
            raise ValueError(f'evaluation user {u} has no test items')
        items = ranked.get(u, [])
        row = {'user_idx': u, 'n_recs': len(items)}
        for k in ks:
            row[f'precision@{k}'] = precision_at_k(items, relevant[u], k)
            row[f'recall@{k}'] = recall_at_k(items, relevant[u], k)
            row[f'ndcg@{k}'] = ndcg_at_k(items, relevant[u], k)
        rows.append(row)
    return pd.DataFrame(rows)


def metric_columns(ks=KS):
    return [f'{m}@{k}' for k in ks for m in ('precision', 'recall', 'ndcg')]