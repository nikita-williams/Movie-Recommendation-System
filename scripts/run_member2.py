
"""Run Member 2 popularity and Exact Jaccard on real MovieLens data.

Run from the project root:
    python -m scripts.run_member2 --scale 10 --users 5
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from src.popularity import popular_recommendations
from src.exact_jaccard import exact_neighbors
from src.scoring import score_recommendations
from src.timing import measure


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=int, choices=[10, 30, 100],
                        default=10)
    parser.add_argument("--users", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--neighbors", type=int, default=50)
    args = parser.parse_args()

    if args.users < 1 or args.top_k < 1 or args.neighbors < 1:
        parser.error("users, top-k and neighbors must be positive")

    data = Path("data/processed")
    output = Path("results/member2")
    output.mkdir(parents=True, exist_ok=True)

    records = []

    # Load the matrices and MovieLens ID mapping
    with measure(records, "load_data"):
        item_user = sparse.load_npz(
            data / f"item_user_{args.scale}.npz"
        ).tocsr()

        user_item = item_user.T.tocsr()
        movie_ids = np.load(data / f"items_{args.scale}.npy")
        eval_users = np.load(data / "eval_users.npy")

        history = pd.read_csv(data / "eval_train_history.csv")
        history = history.sort_values(
            ["user_idx", "timestamp"],
            ascending=[True, False],
            kind="stable"
        )

    if len(movie_ids) != item_user.shape[0]:
        raise ValueError("Movie mapping does not match matrix")

    if user_item.shape[0] <= int(np.max(eval_users)):
        raise ValueError("Evaluation user index exceeds matrix size")

    # MovieLens movieId -> matrix row
    movie_to_row = {
        int(movie_id): row
        for row, movie_id in enumerate(movie_ids)
    }

    histories = {
        int(user): group
        for user, group in history.groupby("user_idx")
    }

    # Cache Exact Jaccard results so repeated seeds are computed once
    neighbor_cache = {}

    popularity_rows = []
    exact_rows = []

    selected_users = eval_users[:args.users]

    print(f"\nDataset scale: {args.scale}%")
    print(f"Matrix shape: {item_user.shape}")
    print(f"Users to evaluate: {len(selected_users)}")

    for position, user in enumerate(selected_users, 1):
        user = int(user)

        if user not in histories:
            raise ValueError(f"Missing history for user {user}")

        user_history = histories[user]

        # All catalog movies in the user's training history
        seen_rows = {
            movie_to_row[int(movie_id)]
            for movie_id in user_history["movieId"]
            if int(movie_id) in movie_to_row
        }

        # Positive training items in the binary matrix
        positive_rows = set(map(int, user_item.getrow(user).indices))

        # Up to 20 most recent positive training items
        query_items = []
        for movie_id in user_history["movieId"]:
            row = movie_to_row.get(int(movie_id))
            if row is None or row not in positive_rows:
                continue
            if row not in query_items:
                query_items.append(row)
            if len(query_items) == 20:
                break

        if not query_items:
            raise ValueError(
                f"No recent positive training items for user {user}"
            )

        # 1. Popularity baseline
        with measure(records, f"popularity_user_{user}"):
            pop_ids, pop_scores = popular_recommendations(
                user_item, user, k=args.top_k
            )

        # Exclude all training-history movies, not only positive ones
        pop_pairs = [
            (int(idx), float(score))
            for idx, score in zip(pop_ids, pop_scores)
            if int(idx) not in seen_rows
        ]

        # 2. Exact Jaccard neighbours
        with measure(records, f"exact_user_{user}"):
            lookup = {}

            for query in query_items:
                if query not in neighbor_cache:
                    neighbor_cache[query] = exact_neighbors(
                        item_user,
                        user_item,
                        query,
                        top_n=args.neighbors
                    )
                lookup[query] = neighbor_cache[query]

            exact_ids, exact_scores = score_recommendations(
                query_items,
                seen_rows,
                lookup,
                k=args.top_k
            )

        # Store recommendations using real MovieLens movie IDs
        for rank, (idx, score) in enumerate(pop_pairs, 1):
            popularity_rows.append({
                "user_idx": user,
                "movieId": int(movie_ids[idx]),
                "rank": rank,
                "score": score,
                "method": "popularity"
            })

        for rank, (idx, score) in enumerate(
            zip(exact_ids, exact_scores), 1
        ):
            exact_rows.append({
                "user_idx": user,
                "movieId": int(movie_ids[idx]),
                "rank": rank,
                "score": float(score),
                "method": "exact_jaccard"
            })

        print(
            f"User {position}/{len(selected_users)} "
            f"(ID {user}): popularity={len(pop_pairs)}, "
            f"exact={len(exact_ids)}"
        )

    prefix = f"scale{args.scale}_users{len(selected_users)}"

    pd.DataFrame(
        popularity_rows,
        columns=["user_idx", "movieId", "rank", "score", "method"]
    ).to_csv(output / f"popularity_{prefix}.csv", index=False)

    pd.DataFrame(
        exact_rows,
        columns=["user_idx", "movieId", "rank", "score", "method"]
    ).to_csv(output / f"exact_jaccard_{prefix}.csv", index=False)

    with open(output / f"timing_{prefix}.json", "w") as file:
        json.dump(records, file, indent=2)

    print("\nCOMPLETE")
    print("Output directory:", output)
    print("Popularity recommendations:", len(popularity_rows))
    print("Exact Jaccard recommendations:", len(exact_rows))
    print("Unique neighbour queries:", len(neighbor_cache))


if __name__ == "__main__":
    main()
