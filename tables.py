
import argparse
from pathlib import Path
import pandas as pd
from srcc.metrics import KS







NAMES = {'popularity': 'Popularity', 'exact_jaccard': 'Exact Jaccard', 'lsh_64_2': 'LSH (64, 2)',
         'lsh_32_4': 'LSH (32, 4)', 'lsh_16_8': 'LSH (16, 8)', 'lsh_8_16': 'LSH (8, 16)'}


def write_table(df, folder, name, caption):
    head = list(df.columns)
    body = [[str(v) for v in row] for row in df.itertuples(index=False)]
    md = [f'**{caption}**', '', '| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    md += ['| ' + ' | '.join(row) + ' |' for row in body]
    (folder / f'{name}.md').write_text('\n'.join(md) + '\n', encoding='utf-8')

    def tex(s):
        for old, new in (('%', r'\%'), ('_', r'\_'), ('±', r'$\pm$'), ('≥', r'$\geq$'), ('×', r'$\times$')):
            s = s.replace(old, new)
        return s
    lines = [r'\begin{table}[ht]\centering', rf'\caption{{{tex(caption)}}}',
             r'\begin{tabular}{' + 'l' * len(head) + '}', r'\hline',
             ' & '.join(map(tex, head)) + r' \\', r'\hline']
    lines += [' & '.join(map(tex, row)) + r' \\' for row in body]
    lines += [r'\hline', r'\end{tabular}', r'\end{table}']
    (folder / f'{name}.tex').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def quality_table(summary, k):
    rows = []
    for (scale, method), g in summary.groupby(['scale', 'method'], sort=False):
        g = g.set_index('metric')
        row = {'Scale': f'{scale}%', 'Method': NAMES[method]}
        for label, metric in (('Precision', 'precision'), ('Recall', 'recall'), ('NDCG', 'ndcg')):
            v = g.loc[f'{metric}@{k}']
            row[f'{label}@{k} [95% CI]'] = f"{v['mean']:.4f} [{v.ci_low:.4f}, {v.ci_high:.4f}]"
        rows.append(row)
    return pd.DataFrame(rows)


def candidate_table(c):
    return pd.DataFrame({
        'Scale': c.scale.map(lambda s: f'{s}%'), 'Method': c.method.map(NAMES),
        'Candidate recall': c.candidate_recall_mean.map('{:.4f}'.format),
        'Mean candidates': c.candidate_size_mean.map('{:.3f}'.format),
        'LSH ms/query': c.lsh_ms_per_query_median_rep.map('{:.3f}'.format),
        'Exact ms/query': c.exact_ms_per_query_reference.map('{:.3f}'.format),
        'Speed-up': c.speedup_vs_exact.map('{:.1f}×'.format),
        'Index MB': (c.signature_mb + c.index_mb).map('{:.1f}'.format)})


def viability_table(v):
    def yes_no(x):
        return {'True': 'yes', 'False': 'no'}.get(str(x), str(x))
    return pd.DataFrame({
        'Scale': v.scale.map(lambda s: f'{s}%'), 'Method': v.method.map(NAMES),
        'Candidate recall': v.candidate_recall.map('{:.4f}'.format),
        'Recall ≥ 0.80': v.recall_ok.map(yes_no),
        'NDCG@10 / exact [95% CI]': [
            'not measured' if pd.isna(r.ndcg10_ratio_to_exact) else
            f'{r.ndcg10_ratio_to_exact:.3f} [{r.ratio_ci_low:.3f}, {r.ratio_ci_high:.3f}]' for r in v.itertuples()],
        'Within 5%': v.ndcg_ok.map(yes_no), 'Verdict': v.verdict})


def timing_table(t):
    g = t.groupby(['scale', 'method'], sort=False)
    out = g.agg(mean=('mean_ms_per_user', 'median'), p95=('p95_ms_per_user', 'median'),
                rss=('max_process_rss_mb', 'max')).reset_index()
    return pd.DataFrame({'Scale': out.scale.map(lambda s: f'{s}%'), 'Method': out.method.map(NAMES),
                         'Mean ms/user': out['mean'].map('{:.2f}'.format),
                         'p95 ms/user': out.p95.map('{:.2f}'.format),
                         'Max process RSS MB': out.rss.map('{:.0f}'.format)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default='results/final')
    a = ap.parse_args()
    src = Path(a.input)
    out = src / 'tables'
    out.mkdir(parents=True, exist_ok=True)
    note = ' (preliminary)' if src.name == 'preliminary' else ''
    write_table(candidate_table(pd.read_csv(src / 'candidate_stats.csv')), out, 'candidates',
                f'LSH candidate recall and cost, mean of 3 hash seeds{note}')
    write_table(viability_table(pd.read_csv(src / 'viability.csv')), out, 'viability',
                f'Viability decision per LSH setting{note}')
    if (src / 'summary_metrics.csv').exists():
        summary = pd.read_csv(src / 'summary_metrics.csv')
        for k in KS:
            write_table(quality_table(summary, k), out, f'quality_at_{k}', f'Recommendation quality at K = {k}')
        write_table(timing_table(pd.read_csv(src / 'timing_baselines.csv')), out, 'timing_baselines',
                    'Baseline latency per user, median of 3 repetitions')
    print('wrote tables to', out)


if __name__ == '__main__':
    main()