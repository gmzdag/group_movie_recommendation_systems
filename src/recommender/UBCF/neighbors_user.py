import time
import pickle
import os
import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial

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

def _worker(uid, R, sim_fn, K):
    """
    Worker function for parallel processing.
    Needs to handle the fact that R (DataFrame) might be large to pickle, 
    but for this scale it might be okay or relies on copy-on-write.
    """
    return uid, compute_neighbors_for_user(R, sim_fn, uid, K)

def precompute_all_user_neighbors(R, sim_fn, K=25):
    users = list(R.index)
    total = len(users)
    neighbors = {}

    start = time.time()
    
    # Use serial if small dataset to avoid overhead
    if total < 200:
        print(f"[SERIAL] Computing neighbors for {total} users (Small N)...")
        for idx, uid in enumerate(users):
            neighbors[uid] = compute_neighbors_for_user(R, sim_fn, uid, K)
            if (idx + 1) % 10 == 0:
                print(f"[SERIAL] {idx+1}/{total}")
                
        print(f"[DONE] neighbors computed in {time.time()-start:.1f}s")
        return neighbors

    # Determine chunksize and pool size
    num_processes = min(cpu_count(), 8) # Cap at 8 to be safe
    print(f"[PARALLEL] Computing neighbors with {num_processes} processes...")
    
    func = partial(_worker, R=R, sim_fn=sim_fn, K=K)
    
    with Pool(processes=num_processes) as pool:
        # Use imap_unordered to track progress
        results_iter = pool.imap_unordered(func, users, chunksize=10)
        
        for idx, result in enumerate(results_iter):
            uid, neigh = result
            neighbors[uid] = neigh
            
            if (idx + 1) % 100 == 0 or (idx + 1) == total:
                print(f"[PARALLEL] Computed {idx + 1}/{total} users ({((idx+1)/total)*100:.1f}%)")
    
    print(f"[DONE] neighbors computed in {time.time()-start:.1f}s")
    return neighbors


# ----------- CACHE UTILS -----------

def save_neighbors(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
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
    # This is single user, no need for parallel
    neighbors[new_uid] = compute_neighbors_for_user(R, sim_fn, new_uid, K)
    save_neighbors(cache_path, neighbors)
    print("[INCREMENTAL] Updated & saved cache.")
    return neighbors
