

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


"""

Grid: scales 10/30/100 x methods (popularity, exact Jaccard, 4 LSH settings) x 3 MinHash seeds x
3 timing repetitions. Popularity and exact Jaccard use no hashing, so they have no seed: they are
repeated 3 times for timing only.

"""


from srcc.bootstrap import mean_ci, paired_ci
from srcc.metrics import KS, evaluate_users, metric_columns

ROOT = Path(__file__).resolve().parents[1]
SCALES = [10, 30, 100]
SEEDS = [0, 1, 2]
REPS = 3
SETTINGS = [(64, 2), (32, 4), (16, 8), (8, 16)]
TOP_K = max(KS)
BASELINES = ['popularity', 'exact_jaccard']
METHODS = BASELINES + [f'lsh_{b}_{r}' for b, r in SETTINGS]
MIN_CANDIDATE_RECALL = 0.80
MAX_NDCG_DROP = 0.05  # relative: LSH NDCG-10 must be >= 95% of exact NDCG@10


def _call(args, cwd):
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    print('>>', ' '.join(args), flush=True)
    log = Path(cwd) / 'run.log'
    with open(log, 'a') as f:
        done = subprocess.run([sys.executable, '-m', *args], cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT)
    if done.returncode:
        raise RuntimeError(f'run failed, see {log}:\n' + log.read_text()[-3000:])


def run_experiments(data, runs, scales, n_users):
    for scale in scales:
        for rep in range(1, REPS + 1):
            work = runs / f'scale{scale}' / f'{rep}'
            if work.exists():
                shutil.rmtree(work)
            (work / 'data' / 'processed').mkdir(parents=True)
            for name in (f'item_user_{scale}.npz', f'items_{scale}.npy', 'eval_users.npy',
                         'eval_train_history.csv'):
                shutil.copy2(data / name, work / 'data' / 'processed' / name)
            _call(['scripts.run', '--scale', str(scale), '--users', str(n_users),
                   '--top-k', str(TOP_K)], work)
            shutil.rmtree(work / 'data')
        out = runs / f'scale{scale}' / 'member3'
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)
        _call(['scripts.run', '--scale', str(scale), '--users', str(n_users), '--top-k', str(TOP_K),
               '--seeds', *map(str, SEEDS), '--reps', str(REPS),
               '--data', str(data.resolve()), '--out', str(out.resolve())], out)


def _read_recs(path, method):
    if not path.exists():
        raise FileNotFoundError(f'missing result file: {path}')
    recs = pd.read_csv(path)
    if len(recs) and set(recs['method']) != {method}:
        raise ValueError(f'{path} does not contain only method {method}')
    return recs


def load_candidate_stats(folders, scales, n_users):
    
    rows = []
    for scale in scales:
        for b, r in SETTINGS:
            for seed in SEEDS:
                path = folders[scale] / f'lsh_candidate_stats_b{b}_r{r}_scale{scale}_seed{seed}.json'
                stats = json.loads(path.read_text())
                if stats['n_users_evaluated'] != n_users:
                    raise ValueError(f'{path}: {stats["n_users_evaluated"]} users, expected {n_users}')
                stats.pop('rep_total_seconds')
                stats.pop('notes')
                rows.append(stats)
    return pd.DataFrame(rows)


def summarise_candidates(stats):
    cols = ['candidate_recall_mean', 'candidate_size_mean', 'candidate_fraction_of_items_mean',
            'share_queries_zero_recall', 'lsh_ms_per_query_median_rep', 'exact_ms_per_query_reference',
            'signature_build_s', 'index_build_s', 'signature_mb', 'index_mb']
    grouped = stats.groupby(['scale', 'method'], sort=False)
    out = grouped[cols].mean()
    out['candidate_recall_seed_min'] = grouped['candidate_recall_mean'].min()
    out['candidate_recall_seed_max'] = grouped['candidate_recall_mean'].max()
    # ratio of the mean times (not the mean of per-seed ratios)
    out['speedup_vs_exact'] = out['exact_ms_per_query_reference'] / out['lsh_ms_per_query_median_rep']
    return out.reset_index()


def evaluate_scale(scale, runs, test, eval_users):
    """Per-user metrics for every method at one scale, averaged over seeds for LSH."""
    folder = runs / f'scale{scale}'
    n = len(eval_users)
    frames, timing = [], []
    for method in BASELINES:
        reference = None
        for rep in range(1, REPS + 1):
            result_dir = folder / f'{rep}' / 'results' / ''
            recs = _read_recs(result_dir / f'{method}_scale{scale}_users{n}.csv', method)
            key = recs[['user_idx', 'movieId', 'rank']].reset_index(drop=True)
            if reference is None:
                reference, per_user = key, evaluate_users(recs, test, eval_users)
            elif not key.equals(reference):
                raise ValueError(f'{method} scale {scale}: recommendations differ between repetitions')
            records = json.loads((result_dir / f'timing_scale{scale}_users{n}.json').read_text())
            label = 'popularity_user_' if method == 'popularity' else 'exact_user_'
            ms = [1000 * t['elapsed_s'] for t in records if t['label'].startswith(label)]
            if len(ms) != n:
                raise ValueError(f'expected {n} {label} timings, found {len(ms)}')
            timing.append({'scale': scale, 'method': method, 'rep': rep,
                           'mean_ms_per_user': float(np.mean(ms)),
                           'p95_ms_per_user': float(np.percentile(ms, 95)),
                           'max_process_rss_mb': max(t['rss_after_mb'] for t in records)})
        per_user.insert(0, 'method', method)
        frames.append(per_user)
    for b, r in SETTINGS:
        method = f'lsh_{b}_{r}'
        seeds = []
        for seed in SEEDS:
            path = folder / '' / f'lsh_b{b}_r{r}_scale{scale}_seed{seed}_users{n}.csv'
            seeds.append(evaluate_users(_read_recs(path, method), test, eval_users))
        mean = pd.concat(seeds).groupby('user_idx', sort=False).mean().reset_index()
        mean.insert(0, 'method', method)
        frames.append(mean)
    out = pd.concat(frames, ignore_index=True)
    out.insert(0, 'scale', scale)
    return out, timing


def summarise_metrics(per_user):
    summary, comparison = [], []
    for scale, at_scale in per_user.groupby('scale', sort=False):
        exact = at_scale[at_scale.method == 'exact_jaccard'].set_index('user_idx')
        for method, g in at_scale.groupby('method', sort=False):
            g = g.set_index('user_idx').loc[exact.index]
            for metric in metric_columns():
                mean, low, high = mean_ci(g[metric].to_numpy())
                summary.append({'scale': scale, 'method': method, 'metric': metric, 'mean': mean,
                                'ci_low': low, 'ci_high': high, 'n_users': len(g)})
                if method != 'exact_jaccard':
                    comparison.append({'scale': scale, 'method': method, 'metric': metric,
                                       **paired_ci(g[metric].to_numpy(), exact[metric].to_numpy())})
    return pd.DataFrame(summary), pd.DataFrame(comparison)


def decide_viability(candidates, comparison=None):
    """A setting passes at a scale only if BOTH rules hold. Without NDCG results it can still FAIL
    (recall rule broken) but can never PASS: the verdict is then 'undecided'."""
    rows = []
    for c in candidates.itertuples():
        recall_ok = bool(c.candidate_recall_mean >= MIN_CANDIDATE_RECALL)
        ratio = low = high = np.nan
        if comparison is not None:
            m = comparison[(comparison.scale == c.scale) & (comparison.method == c.method)
                           & (comparison.metric == 'ndcg@10')].iloc[0]
            ratio, low, high = m.ratio, m.ratio_low, m.ratio_high
        ndcg_ok = None if np.isnan(ratio) else bool(ratio >= 1 - MAX_NDCG_DROP)
        if not recall_ok or ndcg_ok is False:
            verdict = 'FAIL'
        elif ndcg_ok is None:
            verdict = 'undecided'
        else:
            verdict = 'PASS'
        rows.append({'scale': c.scale, 'method': c.method, 'candidate_recall': c.candidate_recall_mean,
                     'recall_ok': recall_ok, 'ndcg10_ratio_to_exact': ratio, 'ratio_ci_low': low,
                     'ratio_ci_high': high, 'ndcg_ok': 'not measured' if ndcg_ok is None else ndcg_ok,
                     'verdict': verdict})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', default='data/processed')
    ap.add_argument('--results', default='results')
    ap.add_argument('--scales', type=int, nargs='+', choices=SCALES, default=SCALES)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument('--skip-run', action='store_true')
    mode.add_argument('--candidates-only', action='store_true')
    a = ap.parse_args()
    data, results = Path(a.data).resolve(), Path(a.results).resolve()

    if a.candidates_only:
        folder = results / ''
        first = next(folder.glob('lsh_candidate_stats_*.json'), None)
        if first is None:
            ap.error(f'no candidate stats in {folder}')
        n_users = json.loads(first.read_text())['n_users_evaluated']
        stats = load_candidate_stats({s: folder for s in a.scales}, a.scales, n_users)
        out = results / 'preliminary'
        out.mkdir(parents=True, exist_ok=True)
        candidates = summarise_candidates(stats)
        candidates.to_csv(out / 'candidate_stats.csv', index=False)
        verdicts = decide_viability(candidates)
        verdicts.to_csv(out / 'viability.csv', index=False)
        print(verdicts.to_string(index=False))
        return

    for name in ('eval_users.npy', 'eval_test.csv'):
        if not (data / name).exists():
            ap.error(f'{data / name} not found. \'s preprocess.py first.')
    eval_users = np.load(data / 'eval_users.npy')
    test = pd.read_csv(data / 'eval_test.csv')
    runs = results / 'runs'
    runs.mkdir(parents=True, exist_ok=True)
    if not a.skip_run:
        run_experiments(data, runs, a.scales, len(eval_users))

    per_user, timing = [], []
    for scale in a.scales:
        frame, times = evaluate_scale(scale, runs, test, eval_users)
        per_user.append(frame)
        timing += times
    per_user = pd.concat(per_user, ignore_index=True)
    stats = load_candidate_stats({s: runs / f'scale{s}' / '' for s in a.scales}, a.scales, len(eval_users))
    candidates = summarise_candidates(stats)
    summary, comparison = summarise_metrics(per_user)
    verdicts = decide_viability(candidates, comparison)

    final = results / 'final'
    final.mkdir(parents=True, exist_ok=True)
    for name, frame in [('per_user_metrics', per_user), ('summary_metrics', summary),
                        ('comparison_vs_exact', comparison), ('candidate_stats', candidates),
                        ('candidate_stats_by_seed', stats), ('timing_baselines', pd.DataFrame(timing)),
                        ('viability', verdicts)]:
        frame.to_csv(final / f'{name}.csv', index=False)
    print(verdicts.to_string(index=False))
    print(f'\nCSV files are in {final}\nNext: python -m src.tables && python -m src.figures')


if __name__ == '__main__':
    main()