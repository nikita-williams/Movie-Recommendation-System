"""Portable timing and process-memory helpers (RSS is process-wide, not per-function)."""
import time
from contextlib import contextmanager
import psutil


@contextmanager
def measure(records, label):
    """Append elapsed seconds and RSS before/after to mutable records list."""
    process = psutil.Process()
    before = process.memory_info().rss
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        after = process.memory_info().rss
        records.append({'label': label, 'elapsed_s': elapsed,
                        'rss_before_mb': before / 2**20,
                        'rss_after_mb': after / 2**20})
