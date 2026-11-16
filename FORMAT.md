# Data interface (Member 1 -> everyone)

Produced by `python preprocess.py` into `data/processed/`.

Scales are tagged `10`, `30`, `100` (percent of users) and are nested: the 10% users are
inside the 30% users, which are inside the 100%. Row indices are per scale; map them back
to movieIds with `items_<tag>.npy`.

| File                             | Contents                                                                                                                                                      |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `item_user_<tag>.npz`            | scipy CSR, shape (n_items_kept, n_users_total). Row = item, column = user index, 1 = the user positively rated (>= 4.0) the item before the cutoff            |
| `train_matrix_by_user_<tag>.npz` | transpose of the above, shape (n_users_total, n_items_kept)                                                                                                   |
| `items_<tag>.npy`                | original `movieId` for each row of `item_user_<tag>`                                                                                                          |
| `user_ids.npy`                   | original `userId` for each user index                                                                                                                         |
| `eval_users.npy`                 | the fixed evaluation user indices (seeded, same for all methods and scales). 580 users on the full data (up to 1,000 if enough are eligible)                  |
| `eval_train_history.csv`         | `user_idx, movieId, timestamp`: training positives of the evaluation users, sorted by time. Take the last 20 whose movieId is in `items_<tag>` as query items |
| `eval_test.csv`                  | `user_idx, movieId`: held-out positives (after the cutoff) of the evaluation users, restricted to items kept at the 10% scale                                 |
| `data_stats.json`                | cutoff, per-scale sizes, evaluation-user counts, a few dataset statistics                                                                                     |
| `movies_subset.csv`              | movieId, title, genres for the items kept at 100%                                                                                                             |

Settings: seed 612; positive = rating >= 4.0; about 15% of ratings after one global cutoff;
an item needs >= 10 positive training ratings (per scale); an evaluation user needs >= 10
positive training ratings and >= 1 held-out positive, both on items kept at the 10% scale.

## Load example

```python
import numpy as np, scipy.sparse as sp, pandas as pd
d = "data/processed/"
item_user = sp.load_npz(d + "item_user_10.npz")  # (items, users)
items = np.load(d + "items_10.npy")              # row -> movieId
eval_users = np.load(d + "eval_users.npy")
hist = pd.read_csv(d + "eval_train_history.csv")
test = pd.read_csv(d + "eval_test.csv")
```

### Data Stats

// data-stats.json

```
{
  "rows": 25000095,
  "missing_values": 0,
  "duplicate_pairs": 0,
  "cutoff_date": "2017-03-24 01:50:04",
  "share_after_cutoff": 0.15000002999988601,
  "scales": {
    "10": {
      "users": 16255,
      "movies_kept": 6135,
      "likes": 1015439
    },
    "30": {
      "users": 48763,
      "movies_kept": 9161,
      "likes": 3149794
    },
    "100": {
      "users": 162541,
      "movies_kept": 12870,
      "likes": 10559313
    }
  },
  "eval": {
    "users_drawn": 580,
    "eligible_users": 580,
    "share_of_10pct_users_eligible": 0.035681328821900955
  },
  "full_data": {
    "users": 162541,
    "movies": 59047,
    "positive_share": 0.4981105471799207,
    "top_1pct_movies_share_of_ratings": 0.4759133115294162
  }
}
```
