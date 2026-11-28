import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from srcc.metrics import KS
from srcc.tables import NAMES





"""Reads run_all's CSV files, writes PDF plots.

    python -m srcc.figures
    python -m srcc.figures --input results/preliminary --out figures/preliminary
"""


# colour-blind-safe colours; markers and line styles so the plots also work in black and white
STYLE = {'popularity': ('#555555', 's', '--'), 'exact_jaccard': ('#000000', 'o', '-'),
         'lsh_64_2': ('#0072B2', '^', '-'), 'lsh_32_4': ('#D55E00', 'v', '-'),
         'lsh_16_8': ('#009E73', 'D', ':'), 'lsh_8_16': ('#CC79A7', 'x', '-.')}
SCALES = [10, 30, 100]


def _finish(fig, ax, path, ncol=3):
    ax.grid(axis='y', color='#dddddd', lw=0.7)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.17), ncol=ncol, frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


def _lines(df, value, ylabel, path, low=None, high=None, hline=None, hlabel=None, log=False):
    """One line per method across the three dataset scales."""
    fig, ax = plt.subplots(figsize=(6, 4.2))
    scales = [s for s in SCALES if s in set(df.scale)]
    methods = [m for m in STYLE if m in set(df.method)]
    offsets = np.linspace(-0.12, 0.12, len(methods)) if low else np.zeros(len(methods))
    for method, off in zip(methods, offsets):
        d = df[df.method == method].set_index('scale').loc[scales]
        x = np.arange(len(scales)) + off
        colour, marker, line = STYLE[method]
        err = None if low is None else np.vstack([d[value] - d[low], d[high] - d[value]]).clip(min=0)
        ax.errorbar(x, d[value], yerr=err, color=colour, marker=marker, ls=line, lw=1.6, ms=5,
                    capsize=2, label=NAMES[method])
    if hline is not None:
        ax.axhline(hline, color='#b00020', lw=1.2, ls='--', label=hlabel)
    if log:
        ax.set_yscale('log')
    elif hline is None:
        ax.set_ylim(bottom=0)
    else:
        ax.set_ylim(-0.02, 1.02)
    ax.set_xticks(range(len(scales)), [f'{s}%' for s in scales])
    ax.set_xlabel('Dataset scale (share of users)')
    ax.set_ylabel(ylabel)
    _finish(fig, ax, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default='results/final')
    ap.add_argument('--out', default='figures')
    a = ap.parse_args()
    src, out = Path(a.input), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    cand = pd.read_csv(src / 'candidate_stats.csv')
    _lines(cand, 'candidate_recall_mean', 'Candidate recall (top-50, 3-seed mean)',
           out / 'candidate_recall.pdf', hline=0.80, hlabel='Viability threshold (0.80)')
    latency = cand[['scale', 'method', 'lsh_ms_per_query_median_rep']].rename(
        columns={'lsh_ms_per_query_median_rep': 'ms'})
    # exact search was timed once per Member 3 run (same process as LSH), so it is the fair reference
    exact = cand.groupby('scale', as_index=False)['exact_ms_per_query_reference'].first()
    exact = exact.rename(columns={'exact_ms_per_query_reference': 'ms'}).assign(method='exact_jaccard')
    _lines(pd.concat([latency, exact]), 'ms', 'Neighbour search time per query movie (ms, log scale)',
           out / 'query_latency.pdf', log=True)

    if (src / 'summary_metrics.csv').exists():
        summary = pd.read_csv(src / 'summary_metrics.csv')
        for k in KS:
            for metric, label in (('precision', 'Precision'), ('recall', 'Recall'), ('ndcg', 'NDCG')):
                _lines(summary[summary.metric == f'{metric}@{k}'], 'mean', f'{label}@{k} (95% CI)',
                       out / f'{metric}_at_{k}.pdf', low='ci_low', high='ci_high')
        # trade-off: quality kept versus candidate recall, with both viability limits
        v = pd.read_csv(src / 'viability.csv')
        fig, ax = plt.subplots(figsize=(6, 4.2))
        for method, g in v.groupby('method', sort=False):
            colour, marker, _ = STYLE[method]
            ax.scatter(g.candidate_recall, g.ndcg10_ratio_to_exact, color=colour, marker=marker,
                       s=40, label=NAMES[method])
        ax.axvline(0.80, color='#b00020', ls='--', lw=1.2, label='Viability limits')
        ax.axhline(0.95, color='#b00020', ls='--', lw=1.2)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(bottom=0, top=max(1.1, np.nanmax(v.ndcg10_ratio_to_exact) * 1.1))
        ax.set_xlabel('Candidate recall (one point per scale)')
        ax.set_ylabel('NDCG@10 as a share of exact Jaccard')
        _finish(fig, ax, out / 'viability.pdf')
    print('wrote figures to', out)


if __name__ == '__main__':
    main()