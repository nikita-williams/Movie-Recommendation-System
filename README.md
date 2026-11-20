## Member 2 — Popularity Baseline and Exact Jaccard Similarity

**Contributor:** Riyaz  
**Project:** Movie Recommendation System — ENSF 612, University of Calgary  
**Dataset:** MovieLens 25M

### Overview

Member 2 implements two recommendation baselines:

1. **Popularity Baseline:** Recommends the most popular movies that a user has not previously interacted with in the training data.
2. **Exact Jaccard Similarity:** Identifies similar movies using overlapping positive user interactions and produces personalized recommendations.

The implementation also provides a shared recommendation scoring function, runtime and memory measurement utilities, automated testing, and a command-line runner.

### Files Implemented

| File | Purpose |
|---|---|
| `src/popularity.py` | Popularity-based top-K recommendations |
| `src/exact_jaccard.py` | Exact item-to-item Jaccard similarity using sparse data |
| `src/scoring.py` | Shared similarity-based recommendation scoring |
| `src/timing.py` | Execution-time and process-memory measurements |
| `scripts/run_member2.py` | Runs popularity and Exact Jaccard recommenders |
| `tests/test_real_data.py` | Tests Exact Jaccard using prepared data |
| `results/member2/` | Generated recommendations and benchmarking results |

### 1. Installation

Open Terminal and navigate to the repository:

```bash
cd ~/Documents/Movie-Recommendation-System
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

The Member 2 implementation requires NumPy, SciPy, pandas, and psutil, alongside the project's other dependencies.

### 2. Required Data

Before running Member 2, the preprocessing pipeline must have generated the required files in `data/processed/`.

These include:

- `item_user_10.npz`
- `item_user_30.npz`
- `item_user_100.npz`
- `eval_train_history.csv`
- `eval_users.npy`
- Supporting movie and user index mappings produced by preprocessing

The processed matrices are sparse representations of positive training interactions.

**Important:** The dataset and processed data are not included in this README's execution instructions as downloadable repository assets. Obtain or generate them using the project's data-preparation pipeline.

### 3. Run the Popularity and Exact Jaccard Recommenders

The main entry point is:

```bash
python -m scripts.run_member2 --scale 10 --users 5
```

This runs both algorithms on five evaluation users using the 10% dataset.

#### Run a 50-user benchmark at each scale

```bash
python -m scripts.run_member2 --scale 10 --users 50
python -m scripts.run_member2 --scale 30 --users 50
python -m scripts.run_member2 --scale 100 --users 50
```

#### Run all 580 evaluation users on the 10% dataset

```bash
python -m scripts.run_member2 --scale 10 --users 580
```

The runner prints progress for each user and summarizes the number of recommendations generated.

### 4. Run the Automated Tests

Run the real-data Exact Jaccard test:

```bash
python -m pytest tests/test_real_data.py -v
```

Run all discovered project tests:

```bash
python -m pytest -v
```

The real-data Exact Jaccard test has been executed successfully.

### 5. Recommendation Algorithms

#### Popularity Baseline

The popularity baseline counts positive training interactions for each movie, excludes previously seen movies, and ranks eligible movies by popularity.

Implementation:

`src/popularity.py`

Main function:

```python
popular_recommendations(user_item, user_id, k=10)
```

#### Exact Jaccard Similarity

Exact Jaccard measures similarity between two movies based on the users who positively interacted with each movie.

\[
J(i,j)=\frac{|U_i\cap U_j|}{|U_i\cup U_j|}
\]

Where \(U_i\) and \(U_j\) are the sets of users who positively interacted with movies \(i\) and \(j\).

The implementation uses sparse matrices and an inverted-index approach to identify overlapping users efficiently.

Implementation:

`src/exact_jaccard.py`

#### Shared Recommendation Scoring

The shared scoring function aggregates similarity scores from a user's query movies, excludes previously seen movies, and returns the highest-scoring candidates.

\[
Score(j)=\sum_{i\in Q_u}J(i,j)
\]

Where \(Q_u\) is the set of selected positive training-history movies for user \(u\).

Implementation:

`src/scoring.py`

Main function:

```python
score_recommendations(query_items, seen_items, neighbor_lookup, k=10)
```

### 6. Generated Output Files

All results are written to:

`results/member2/`

For example, running:

```bash
python -m scripts.run_member2 --scale 10 --users 50
```

generates:

```text
results/member2/
├── popularity_scale10_users50.csv
├── exact_jaccard_scale10_users50.csv
└── timing_scale10_users50.json
```

Recommendation CSV columns:

| Column | Description |
|---|---|
| `user_idx` | Evaluation user identifier |
| `movieId` | Original MovieLens movie identifier |
| `rank` | Recommendation rank, starting at 1 |
| `score` | Popularity count or aggregated similarity score |
| `method` | Recommendation method |

Timing JSON records include operation labels, elapsed time, and process RSS before and after measurement.

### 7. Benchmark Results

Each scale was benchmarked using 50 evaluation users.

| Dataset scale | Popularity (ms/user) | Exact Jaccard (ms/user) | Maximum observed RSS (MB) |
|---|---:|---:|---:|
| 10% | 2.383 | 11.707 | 193.50 |
| 30% | 6.876 | 26.547 | 222.67 |
| 100% | 22.559 | 79.275 | 495.23 |

Popularity was faster than Exact Jaccard at every tested scale. Exact Jaccard performs personalized similarity-based ranking at additional computational cost.

Memory values represent process-wide RSS sampled at measurement boundaries, not isolated algorithm memory usage or guaranteed peak memory.

### 8. Recommendation Validation

Both recommendation methods were validated on 580 evaluation users using the 10% dataset.

| Validation | Popularity | Exact Jaccard |
|---|---:|---:|
| Users evaluated | 580 | 580 |
| Recommendations generated | 5,800 | 5,800 |
| Previously seen movies | 0 | 0 |
| Duplicate recommendations | 0 | 0 |
| Users receiving 10 recommendations | 580 | 580 |
| Valid recommendation ranks | Yes | Yes |

These checks validate recommendation output structure and exclusion of movies present in the evaluated training history. They do not establish recommendation accuracy.

### 9. Evaluation and Team Integration

The recommendation CSV files can be used by the evaluation component to calculate metrics such as Precision@10 and Recall@10 against held-out test interactions.

Both methods use the same recommendation output schema, enabling comparisons with the approximate similarity method developed by another team member.

### 10. Limitations

- Exact Jaccard is more computationally expensive than popularity.
- Benchmark timings depend on hardware, data size, query caching, and execution conditions.
- Memory measurements reflect the Python process rather than individual algorithm allocations.
- Recommendation-quality metrics must be calculated separately using held-out test data.
- Processed MovieLens data must be prepared locally before executing the runner.
