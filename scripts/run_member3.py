"""Run Member 3's MinHash + LSH recommender on processed MovieLens data.

Run from the project root, e.g.:
    python -m scripts.run_member3 --scale 10 --users 50
    python -m scripts.run_member3 --scale 10 --users 580 --seeds 0 1 2 --reps 3

Per seed and per (b, r) setting this writes to results/member3/ (names end in _b{b}_r{r}_scale{S}_seed{seed}):
    lsh_<tag>_users{N}.csv             recommendations, same columns as results/member2/*.csv
    lsh_neighbours_<tag>.npz           query_item_ids, neighbour_ids, neighbour_sims (row indices, -1 padded,
                                       same layout as exact_neighbors_for_queries) + candidate_sizes,
                                       candidate_recall per query
    lsh_candidate_stats_<tag>.json     candidate recall / size, latency, memory, build times, exact reference

Query-item and history handling copies scripts/run_member2.py so both methods see identical inputs.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from src.candidates import candidate_recall
from src.exact_jaccard import exact_neighbors
from src.lsh import SETTINGS, LSHIndex, threshold
from src.minhash import minhash_signatures
from src.rescore import Rescorer
from src.scoring import score_recommendations
from src.timing import measure

K_SIG = 128


def parse_setting(text):
    b, r = text.lower().split('x')
    return int(b), int(r)


def user_queries(history, user_item, movie_to_row, user):
    """(query rows, seen rows) for one user, exactly as run_member2.py builds them."""
    seen = {movie_to_row[int(m)] for m in history['movieId'] if int(m) in movie_to_row}
    positive = set(map(int, user_item.getrow(user).indices))
    queries = []
    for movie_id in history['movieId']:
        row = movie_to_row.get(int(movie_id))
        if row is None or row not in positive:
            continue
        if row not in queries:
            queries.append(row)
        if len(queries) == 20:
            break
    return queries, seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scale', type=int, choices=[10, 30, 100], default=10)
    ap.add_argument('--users', type=int, default=5)
    ap.add_argument('--top-k', type=int, default=10)
    ap.add_argument('--neighbors', type=int, default=50)
    ap.add_argument('--seeds', type=int, nargs='+', default=[0], help='MinHash hash-function seeds')
    ap.add_argument('--reps', type=int, default=1, help='timing repetitions of the query phase')
    ap.add_argument('--settings', nargs='+', default=[f'{b}x{r}' for b, r in SETTINGS], help='e.g. 64x2 32x4')
    ap.add_argument('--data', default='data/processed')
    ap.add_argument('--out', default='results/member3')
    ap.add_argument('--save-index', action='store_true', help='also save signatures and buckets (.npy/.npz)')
    a = ap.parse_args()
    settings = [parse_setting(s) for s in a.settings]
    if min(a.users, a.top_k, a.neighbors, a.reps) < 1:
        ap.error('users, top-k, neighbors and reps must be positive')

    data, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    records = []

    with measure(records, 'load_data'):
        item_user = sparse.load_npz(data / f'item_user_{a.scale}.npz').tocsr()
        user_item = item_user.T.tocsr()
        movie_ids = np.load(data / f'items_{a.scale}.npy')
        eval_users = np.load(data / 'eval_users.npy')
        history = pd.read_csv(data / 'eval_train_history.csv')
        history = history.sort_values(['user_idx', 'timestamp'], ascending=[True, False], kind='stable')
    if len(movie_ids) != item_user.shape[0]:
        raise ValueError('Movie mapping does not match matrix')
    movie_to_row = {int(m): row for row, m in enumerate(movie_ids)}
    histories = {int(u): g for u, g in history.groupby('user_idx')}

    selected = [int(u) for u in eval_users[:a.users]]
    per_user = {}
    for user in selected:
        if user not in histories:
            raise ValueError(f'Missing history for user {user}')
        queries, seen = user_queries(histories[user], user_item, movie_to_row, user)
        if not queries:
            raise ValueError(f'No recent positive training items for user {user}')
        per_user[user] = (queries, seen)
    unique_queries = np.array(sorted({q for qs, _ in per_user.values() for q in qs}), dtype=np.int32)
    n_valid = int((np.diff(item_user.indptr) > 0).sum())
    prefix = f'scale{a.scale}'
    print(f'\nScale {a.scale}%  matrix {item_user.shape}  users {len(selected)}  unique query items {len(unique_queries)}')

    # Exact reference (Member 2's code): ground truth for candidate recall and a same-machine latency baseline
    # Timed with the same number of repetitions as LSH (median of the repetition totals), so neither side
    # gets a warm-cache advantage the other lacks.
    exact, exact_rep_totals = {}, []
    for rep in range(a.reps):
        total = 0.0
        for q in unique_queries:
            t = time.perf_counter()
            result = exact_neighbors(item_user, user_item, int(q), top_n=a.neighbors)
            total += time.perf_counter() - t
            if rep == 0:
                exact[int(q)] = result
        exact_rep_totals.append(total)
    exact_ms = 1000 * float(np.median(exact_rep_totals)) / len(unique_queries)
    print(f'exact reference: {exact_ms:.2f} ms/query (median of {a.reps} reps)')

    rescorer = Rescorer(item_user)
    for seed in a.seeds:
        with measure(records, f'signatures_seed{seed}'):
            sig = minhash_signatures(item_user, k=K_SIG, seed=seed)
        sig_build_s = records[-1]['elapsed_s']
        if a.save_index:
            np.save(out / f'signatures_{prefix}_seed{seed}.npy', sig)

        for b, r in settings:
            tag = f'b{b}_r{r}_{prefix}_seed{seed}'
            with measure(records, f'index_{tag}'):
                index = LSHIndex(sig, b, r)
            index_build_s = records[-1]['elapsed_s']
            if a.save_index:
                index.save(out / f'buckets_{tag}.npz')

            rep_totals, lookup_s, rescore_s, neighbours, cand_sizes = [], [], [], {}, {}
            for rep in range(a.reps):
                total = 0.0
                for q in unique_queries:
                    t0 = time.perf_counter()
                    cands = index.candidates(int(q))
                    t1 = time.perf_counter()
                    ids, sims = rescorer.neighbors(int(q), cands, a.neighbors)
                    t2 = time.perf_counter()
                    total += t2 - t0
                    if rep == 0:
                        lookup_s.append(t1 - t0)
                        rescore_s.append(t2 - t1)
                        neighbours[int(q)] = (ids, sims)
                        cand_sizes[int(q)] = len(cands)
                        neighbours[int(q)] += (cands,)
                rep_totals.append(total)

            recall = np.array([candidate_recall(neighbours[int(q)][2], exact[int(q)][0]) for q in unique_queries])
            sizes = np.array([cand_sizes[int(q)] for q in unique_queries])
            lookup = {q: (v[0], v[1]) for q, v in neighbours.items()}

            # Recommendations: the shared scoring function from Member 2, unchanged
            rows = []
            for user in selected:
                queries, seen = per_user[user]
                rec_ids, rec_scores = score_recommendations(queries, seen, lookup, k=a.top_k)
                for rank, (idx, score) in enumerate(zip(rec_ids, rec_scores), 1):
                    rows.append({'user_idx': user, 'movieId': int(movie_ids[idx]), 'rank': rank,
                                 'score': float(score), 'method': f'lsh_{b}_{r}'})
            pd.DataFrame(rows, columns=['user_idx', 'movieId', 'rank', 'score', 'method']).to_csv(
                out / f'lsh_{tag}_users{len(selected)}.csv', index=False)

            width = a.neighbors
            ids_arr = np.full((len(unique_queries), width), -1, dtype=np.int32)
            sims_arr = np.zeros((len(unique_queries), width), dtype=np.float32)
            for row, q in enumerate(unique_queries):
                n, s = lookup[int(q)]
                ids_arr[row, :len(n)], sims_arr[row, :len(s)] = n, s
            np.savez_compressed(out / f'lsh_neighbours_{tag}.npz', query_item_ids=unique_queries,
                                neighbour_ids=ids_arr, neighbour_sims=sims_arr,
                                candidate_sizes=sizes, candidate_recall=recall)

            query_ms = 1000 * float(np.median(rep_totals)) / len(unique_queries)
            stats = {
                'method': f'lsh_{b}_{r}', 'bands': b, 'rows': r, 'k': K_SIG, 'threshold': threshold(b, r),
                'scale': a.scale, 'seed': seed, 'n_items': int(item_user.shape[0]), 'n_users_evaluated': len(selected),
                'n_unique_queries': int(len(unique_queries)), 'reps': a.reps,
                'candidate_recall_mean': float(np.nanmean(recall)),
                'candidate_recall_median': float(np.nanmedian(recall)),
                'share_queries_zero_recall': float(np.mean(recall[~np.isnan(recall)] == 0)),
                'candidate_size_mean': float(sizes.mean()), 'candidate_size_median': float(np.median(sizes)),
                'candidate_size_p95': float(np.percentile(sizes, 95)),
                'candidate_fraction_of_items_mean': float(sizes.mean() / max(n_valid - 1, 1)),
                'lookup_ms_per_query': 1000 * float(np.mean(lookup_s)),
                'rescore_ms_per_query': 1000 * float(np.mean(rescore_s)),
                'lsh_ms_per_query_median_rep': query_ms,
                'rep_total_seconds': rep_totals,
                'exact_ms_per_query_reference': exact_ms,
                'speedup_vs_exact': exact_ms / query_ms if query_ms > 0 else None,
                'signature_build_s': sig_build_s, 'index_build_s': index_build_s,
                'signature_mb': sig.nbytes / 2**20, 'index_mb': index.nbytes() / 2**20,
                'rss_mb_after': records[-1]['rss_after_mb'],
                'notes': 'latency = candidate lookup + exact re-scoring per unique query item, build times excluded; '
                         'exact reference measured in the same process with src/exact_jaccard.py',
            }
            with open(out / f'lsh_candidate_stats_{tag}.json', 'w') as f:
                json.dump(stats, f, indent=2)
            print(f'seed {seed} ({b:>2},{r:>2}) thr {threshold(b, r):.2f}  recall {stats["candidate_recall_mean"]:.3f}  '
                  f'cands {sizes.mean():8.1f} ({100 * stats["candidate_fraction_of_items_mean"]:5.1f}% of items)  '
                  f'{query_ms:7.2f} ms/q vs exact {exact_ms:.2f}  ({stats["speedup_vs_exact"]:.2f}x)')

    with open(out / f'timing_member3_{prefix}_users{len(selected)}.json', 'w') as f:
        json.dump(records, f, indent=2)
    print('\nCOMPLETE  output directory:', out)


if __name__ == '__main__':
    main()
