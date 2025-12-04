import time
import pickle
import os
import numpy as np


def compute_neighbors_for_user(R, sim_fn, uid, K=25):
    target_row = R.loc[uid]
    sims = {}

    for other_uid in R.index:
        if other_uid == uid:
            continue

        sim = sim_fn(target_row, R.loc[other_uid])
        if not np.isnan(sim) and sim > 0:
            sims[other_uid] = sim

    return dict(sorted(sims.items(), key=lambda x: x[1], reverse=True)[:K])


def precompute_all_user_neighbors(R, sim_fn, K=25):
    users = list(R.index)
    total = len(users)
    neighbors = {}

    start = time.time()
    last = start

    for idx, uid in enumerate(users):
        neighbors[uid] = compute_neighbors_for_user(R, sim_fn, uid, K)

        now = time.time()
        if now - last >= 1:
            progress = (idx + 1) / total
            elapsed = now - start
            eta = (elapsed / progress) - elapsed
            print(f"[USER-NEIGH] {idx+1}/{total} ({progress*100:.1f}%) | ETA={eta:.1f}s")
            last = now

    print(f"[DONE] neighbors computed in {time.time()-start:.1f}s")
    return neighbors


# ----------- CACHE UTILS -----------

def save_neighbors(path, obj):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_neighbors(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_or_compute_neighbors(cache_path, R, sim_fn, K=25):
    if os.path.exists(cache_path):
        print(f"[CACHE] Loaded neighbor cache from: {cache_path}")
        return load_neighbors(cache_path)

    print("[CACHE] No cache found. Computing neighbors...")
    neigh = precompute_all_user_neighbors(R, sim_fn, K)
    save_neighbors(cache_path, neigh)
    print("[CACHE] Saved.")
    return neigh


def update_neighbors_for_new_user(cache_path, neighbors, R, sim_fn, new_uid, K=25):
    print(f"[INCREMENTAL] Computing neighbors for NEW user {new_uid}")
    neighbors[new_uid] = compute_neighbors_for_user(R, sim_fn, new_uid, K)
    save_neighbors(cache_path, neighbors)
    print("[INCREMENTAL] Updated & saved cache.")
    return neighbors
