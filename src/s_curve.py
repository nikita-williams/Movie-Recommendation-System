"""S-curve figure (Member 3): theoretical 1 - (1 - s^r)^b next to the measured candidate rate.

Measured: for a random sample of query items, take EVERY other item that shares at least one user with
the query (its true Jaccard s comes from Member 2's exact_neighbors with top_n = all). The pair is a hit
if LSH puts the two items in a common bucket. Pairs are binned by s; the measured point is hits / pairs
in the bin. Pairs with no shared user (s = 0) are left out: they cannot be neighbours.

    python -m src.s_curve --scale 10 --queries 300 --seeds 0 1 2
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from src.exact_jaccard import exact_neighbors
from src.lsh import SETTINGS, LSHIndex, theoretical_probability, threshold
from src.minhash import minhash_signatures

BIN_WIDTH = 0.02
MIN_PAIRS = 30  # bins with fewer pairs are too noisy to plot


def collect_pairs(item_user, user_item, n_queries, seed=0):
    """(query, other, true_jaccard) for every overlapping pair from a seeded sample of query items."""
    rng = np.random.default_rng(seed)
    sizes = np.diff(item_user.indptr)
    pool = np.flatnonzero(sizes > 0)
    queries = np.sort(rng.choice(pool, size=min(n_queries, len(pool)), replace=False))
    q_all, o_all, s_all = [], [], []
    for q in queries:
        ids, sims = exact_neighbors(item_user, user_item, int(q), top_n=item_user.shape[0])
        q_all.append(np.full(len(ids), q, dtype=np.int32))
        o_all.append(ids)
        s_all.append(sims)
    return np.concatenate(q_all), np.concatenate(o_all), np.concatenate(s_all).astype(float)


def measure_s_curve(item_user, user_item, n_queries=300, seeds=(0,), settings=SETTINGS, k=128, pair_seed=0):
    """Binned measured candidate rate per setting, pooled over the hash seeds. Returns a DataFrame."""
    q, o, s = collect_pairs(item_user, user_item, n_queries, pair_seed)
    edges = np.arange(0, 1 + BIN_WIDTH, BIN_WIDTH)
    which = np.minimum(np.digitize(s, edges) - 1, len(edges) - 2)
    hits = {st: np.zeros(len(edges) - 1) for st in settings}
    pairs = np.zeros(len(edges) - 1)
    for seed in seeds:
        sig = minhash_signatures(item_user, k=k, seed=seed)
        for b, r in settings:
            hit = LSHIndex(sig, b, r).is_candidate(q, o)
            hits[(b, r)] += np.bincount(which, weights=hit, minlength=len(edges) - 1)
        pairs += np.bincount(which, minlength=len(edges) - 1)
    pairs_per_setting = pairs  # every seed sees the same pairs
    mid = (edges[:-1] + edges[1:]) / 2
    rows = []
    for b, r in settings:
        for i in range(len(mid)):
            if pairs_per_setting[i] == 0:
                continue
            rows.append({'b': b, 'r': r, 'similarity': mid[i], 'pairs': int(pairs_per_setting[i]),
                         'measured': hits[(b, r)][i] / pairs_per_setting[i],
                         'theory': float(theoretical_probability(mid[i], b, r))})
    return pd.DataFrame(rows)


def plot_s_curve(df, path, title=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    ink, grid, series = '#1f2933', '#d9dde3', '#1b6ca8'
    fig, axes = plt.subplots(2, 2, figsize=(8, 6.4), sharex=True, sharey=True)
    s = np.linspace(0, 1, 400)
    for ax, (b, r) in zip(axes.ravel(), SETTINGS):
        d = df[(df.b == b) & (df.r == r) & (df.pairs >= MIN_PAIRS)]
        se = np.sqrt(d.measured * (1 - d.measured) / d.pairs)
        ax.plot(s, theoretical_probability(s, b, r), color=ink, lw=2, label='Theory', zorder=2)
        ax.errorbar(d.similarity, d.measured, yerr=1.96 * se, fmt='o', ms=4, color=series, ecolor=series,
                    elinewidth=1, capsize=0, label='Measured', zorder=3)
        ax.axvline(threshold(b, r), color=ink, lw=1, ls=':', zorder=1)
        ax.set_title(f'b = {b}, r = {r}   (threshold ≈ {threshold(b, r):.2f})', fontsize=10, color=ink)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(color=grid, lw=0.6)
        ax.set_axisbelow(True)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
    axes[0, 0].legend(loc='lower right', frameon=False, fontsize=9)
    for ax in axes[1]:
        ax.set_xlabel('True Jaccard similarity of the item pair')
    for ax in axes[:, 0]:
        ax.set_ylabel('Share of pairs that become candidates')
    if title:
        fig.suptitle(title, fontsize=11, color=ink)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scale', type=int, choices=[10, 30, 100], default=10)
    ap.add_argument('--queries', type=int, default=300)
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--data', default='data/processed')
    ap.add_argument('--out', default='results/member3')
    a = ap.parse_args()
    item_user = sparse.load_npz(Path(a.data) / f'item_user_{a.scale}.npz').tocsr()
    user_item = item_user.T.tocsr()
    df = measure_s_curve(item_user, user_item, a.queries, tuple(a.seeds))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f's_curve_scale{a.scale}.csv', index=False)
    plot_s_curve(df, out / f's_curve_scale{a.scale}.pdf',
                 f'MinHash LSH candidate rate vs true Jaccard (scale {a.scale}%, {len(a.seeds)} seeds)')
    print('wrote', out / f's_curve_scale{a.scale}.pdf')


if __name__ == '__main__':
    main()
