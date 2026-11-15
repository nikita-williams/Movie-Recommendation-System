"""Preprocessing for MovieLens 25M.

Usage:
  python preprocess.py --ratings data/raw/ml-25m/ratings.csv \
      --movies data/raw/ml-25m/movies.csv --out data/processed

Steps:
  1. Load ratings and check them (missing values, duplicates).
  2. Pick ONE time cutoff so ~15% of all ratings come after it (train = before, test = after).
  3. Keep only "positive" ratings (>= 4.0): Jaccard needs liked / not liked.
  4. Shuffle users with a fixed seed -> 10%, 30%, 100% user groups (nested).
  5. For each group build a movie x user table; drop movies with < 10 likes.
  6. Pick the evaluation users (once, from the 10% group) and save their train/test data.
  7. Save simple statistics for the report.
"""
import argparse
import json
import math
import os

import numpy as np
import pandas as pd
import scipy.sparse as sp

SEED = 612            # fixed, so everyone gets identical results
POSITIVE = 4.0        # rating >= 4.0 counts as "liked"
TEST_SHARE = 0.15     # ~15% of all ratings fall after the cutoff
MIN_LIKES = 10        # movies and users need at least this many training likes
N_EVAL = 1000         # upper limit; fewer are used if fewer users are eligible
SCALES = {"10": 0.10, "30": 0.30, "100": 1.00}


def main(ratings_path, movies_path, out):
    os.makedirs(out, exist_ok=True)

    # 1. load + check
    df = pd.read_csv(ratings_path)  # columns: userId, movieId, rating, timestamp
    stats = {"rows": len(df),
             "missing_values": int(df.isna().sum().sum()),
             "duplicate_pairs": int(df.duplicated(["userId", "movieId"]).sum())}
    df = df.drop_duplicates(["userId", "movieId"])

    # 2. one global cutoff
    cutoff = int(np.quantile(df["timestamp"], 1 - TEST_SHARE))
    stats["cutoff_date"] = str(pd.to_datetime(cutoff, unit="s"))
    stats["share_after_cutoff"] = float((df["timestamp"] > cutoff).mean())

    # users get a compact index 0..n-1 (user_ids.npy maps back to userId)
    user_ids = np.sort(df["userId"].unique())
    n_users = len(user_ids)
    df["user"] = np.searchsorted(user_ids, df["userId"])
    np.save(f"{out}/user_ids.npy", user_ids)

    # 3. positives, split into train / test
    pos = df[df["rating"] >= POSITIVE]
    train = pos[pos["timestamp"] <= cutoff]
    test = pos[pos["timestamp"] > cutoff]

    # 4. nested user groups: the same shuffle is used for every size
    order = np.random.default_rng(SEED).permutation(n_users)
    rank = np.empty(n_users, dtype=int)
    rank[order] = np.arange(n_users)  # position of each user in the shuffle

    # 5. movie x user table for each size
    stats["scales"] = {}
    kept_movies = {}
    for tag, share in SCALES.items():
        in_group = rank[train["user"].to_numpy()] < math.ceil(share * n_users)
        t = train[in_group]
        likes = t["movieId"].value_counts()
        keep = likes[likes >= MIN_LIKES].index
        t = t[t["movieId"].isin(keep)]
        movies = np.sort(keep.to_numpy())
        rows = np.searchsorted(movies, t["movieId"])
        table = sp.csr_matrix((np.ones(len(t), dtype=np.uint8), (rows, t["user"])),
                              shape=(len(movies), n_users))
        sp.save_npz(f"{out}/item_user_{tag}.npz", table)                       # movies x users
        sp.save_npz(f"{out}/train_matrix_by_user_{tag}.npz", table.T.tocsr())  # users x movies
        np.save(f"{out}/items_{tag}.npy", movies)                              # row -> movieId
        kept_movies[tag] = movies
        stats["scales"][tag] = {"users": math.ceil(share * n_users),
                                "movies_kept": len(movies), "likes": int(table.nnz)}

    # 6. evaluation users: from the 10% group; its kept movies stay kept in the bigger groups
    small = kept_movies["10"]
    in_small = rank < math.ceil(SCALES["10"] * n_users)
    train_n = train[train["movieId"].isin(small)].groupby("user").size()
    test_ok = test[test["movieId"].isin(small)]
    test_n = test_ok.groupby("user").size()
    eligible = [u for u in train_n.index[train_n >= MIN_LIKES]
                if in_small[u] and u in test_n.index]
    rng = np.random.default_rng(SEED + 1)
    eval_users = np.sort(rng.choice(eligible, size=min(N_EVAL, len(eligible)), replace=False))
    np.save(f"{out}/eval_users.npy", eval_users)

    train[np.isin(train["user"], eval_users)][["user", "movieId", "timestamp"]] \
        .rename(columns={"user": "user_idx"}).sort_values(["user_idx", "timestamp"]) \
        .to_csv(f"{out}/eval_train_history.csv", index=False)
    test_ok[np.isin(test_ok["user"], eval_users)][["user", "movieId"]] \
        .rename(columns={"user": "user_idx"}).to_csv(f"{out}/eval_test.csv", index=False)

    stats["eval"] = {"users_drawn": len(eval_users),
                     "eligible_users": len(eligible),
                     "share_of_10pct_users_eligible": len(eligible) / int(in_small.sum())}

    # 7. a few numbers for the report
    per_movie = df["movieId"].value_counts()
    stats["full_data"] = {
        "users": n_users, "movies": len(per_movie),
        "positive_share": float((df["rating"] >= POSITIVE).mean()),
        "top_1pct_movies_share_of_ratings":
            float(per_movie.iloc[: max(1, len(per_movie) // 100)].sum() / len(df)),
    }
    with open(f"{out}/data_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    if movies_path:
        m = pd.read_csv(movies_path)
        m[m["movieId"].isin(kept_movies["100"])].to_csv(f"{out}/movies_subset.csv", index=False)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratings", required=True)
    ap.add_argument("--movies", default=None)
    ap.add_argument("--out", default="data/processed")
    a = ap.parse_args()
    main(a.ratings, a.movies, a.out)