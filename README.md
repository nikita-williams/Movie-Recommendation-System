# ENSF 612 — Riyaz (Member 2) baseline implementation

## Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

## Components
- `src/popularity.py`: most-popular unseen positive training items.
- `src/exact_jaccard.py`: inverted-index exact Jaccard, using `np.bincount`.
- `src/scoring.py`: shared scoring of up to 20 query items for exact and LSH.
- `src/timing.py`: wall-clock and RSS snapshots.

## Next integration step
Agree on `FORMAT.md` with teammates. Praise supplies binary CSR matrices and per-user recent positive item lists. Nikita imports `score_recommendations` without rewriting it. Favour runs the experiment grid and metrics.

## Notes
- Exact Jaccard only enumerates co-occurring candidates, not all items.
- Build-time, query-time, and process memory must be reported separately.
- Matrix values must be binary 0/1; use training data only.
- Raw MovieLens data must not be committed to the repository.
