# Proposed interface contract — team must approve before integration

All IDs in matrices and arrays are **contiguous zero-based integer indices**, NOT raw MovieLens IDs.

Input per scale (Praise):
- `data/processed/<scale>/train_matrix_by_user.npz`: binary scipy CSR, users × items.
- `data/processed/<scale>/train_matrix_by_item.npz`: binary scipy CSR, items × users, transpose of above.
- `data/processed/<scale>/user_id_map.csv`: columns `orig_user_id,contiguous_id`.
- `data/processed/<scale>/item_id_map.csv`: columns `orig_item_id,contiguous_id`.
- `data/processed/<scale>/recent_positive_items.npz`: arrays `user_ids` and `item_ids` (2D, up to 20 recent positive training item indices, padded -1). Must be sorted most-recent-first using timestamps.
- `data/processed/sampled_users_1k.npy`: stable **original MovieLens user IDs** so each scale can map them to local contiguous IDs. Alternative: one shared global user-index space across scales; decide with team.

Neighbour output (Riyaz and Nikita): `.npz` with arrays `query_item_ids` (N,), `neighbour_ids` (N,50), `neighbour_sims` (N,50). Padding: ID=-1, similarity=0. Descending similarity, tie-break by ascending ID.

Recommendation output: `.npz` with `user_ids` (N,), `item_ids` (N,K), `scores` (N,K). Padding ID=-1, score=0. K defaults to 10; optionally 20. No previously liked training items in recommendations.

Scoring: `score_recommendations(query_items, seen_items, neighbor_lookup, k=10)`, where `neighbor_lookup[query_item] = (neighbor_ids, neighbour_sims)`; pass up to 20 most recent positive training items.

Popularity: top-K unseen items ranked by positive training frequency; tie-break by ascending contiguous item ID.

Timing: distinguish preprocessing/index-build cost from per-query latency. RSS snapshots are NOT true peak RSS; the evaluation owner must choose a sampling or OS high-water-mark method for peak measurement.

This contract resolves ambiguities in the original split and is a proposal, not yet approved.
