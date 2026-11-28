import numpy as np


"""95% percentile bootstrap confidence intervals over evaluation users .

Users are resampled with replacement. When a method has several hash seeds, each user's values are
averaged over the seeds FIRST, so one user is still one observation (seeds are not extra users).
Comparisons with exact Jaccard are PAIRED: the same resampled users are used for both methods.
The intervals describe uncertainty from which users were sampled, not from the choice of hash seed.
"""


N_BOOT = 10000
SEED = 612


def _resampled_means(columns, n_boot, seed):
    """Means of each column (shape n_users x m) under the same n_boot user resamples."""
    rng = np.random.default_rng(seed)
    n = len(columns)
    out = np.empty((n_boot, columns.shape[1]))
    for start in range(0, n_boot, 1000):
        size = min(1000, n_boot - start)
        idx = rng.integers(0, n, size=(size, n))
        out[start:start + size] = columns[idx].mean(axis=1)
    return out


def _check(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) == 0 or np.isnan(values).any():
        raise ValueError('need a non-empty 1-D array without NaN')
    return values


def mean_ci(values, n_boot=N_BOOT, seed=SEED, level=0.95):
    """(mean, low, high)."""
    values = _check(values)
    means = _resampled_means(values[:, None], n_boot, seed)[:, 0]
    low, high = np.quantile(means, [(1 - level) / 2, (1 + level) / 2])
    return float(values.mean()), float(low), float(high)


def paired_ci(method, baseline, n_boot=N_BOOT, seed=SEED, level=0.95):
    """Method versus baseline on the same users (arrays in the same user order).

    Returns the difference (method - baseline) and the ratio (method / baseline) with intervals.
    The ratio is nan when the baseline mean is 0.
    """
    method, baseline = _check(method), _check(baseline)
    if method.shape != baseline.shape:
        raise ValueError('method and baseline must cover the same users')
    means = _resampled_means(np.column_stack([method, baseline]), n_boot, seed)
    q = [(1 - level) / 2, (1 + level) / 2]
    low, high = np.quantile(means[:, 0] - means[:, 1], q)
    out = {'diff': float(method.mean() - baseline.mean()), 'diff_low': float(low), 'diff_high': float(high),
           'ratio': float('nan'), 'ratio_low': float('nan'), 'ratio_high': float('nan')}
    if baseline.mean() > 0:
        ok = means[:, 1] > 0
        low, high = np.quantile(means[ok, 0] / means[ok, 1], q)
        out.update(ratio=float(method.mean() / baseline.mean()), ratio_low=float(low), ratio_high=float(high))
    return out