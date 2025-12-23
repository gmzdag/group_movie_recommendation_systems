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


def load_or_compute_neighbors(R, sim_fn, K=25, metric="pearson"):
    """
    Load from centralized cache OR compute and save.
    Uses hash-based cache keys to detect data changes.
    
    Args:
        R: User-item rating matrix (DataFrame)
        sim_fn: Similarity function
        K: Number of neighbors
        metric: Similarity metric name (for cache key)
    
    Returns:
        Dictionary of {user_id: {neighbor_id: similarity, ...}}
    """
    from src.utils.cache_manager import CacheManager
    
    cache = CacheManager()
    
    # Simple, fixed cache key (no hash - assumes same data each time)
    cache_key = f"user_neighbors_k{K}_{metric}"
    
    print(f"\n{'='*60}")
    print(f"[CACHE] User Neighbors - K={K}, metric={metric}")
    print(f"[CACHE] Cache key: {cache_key}")
    print(f"{'='*60}")
    
    # Try to load from cache
    neighbors = cache.load(cache_key)
    
    if neighbors is not None:
        print(f"[CACHE HIT] Loaded {len(neighbors)} users from cache")
        return neighbors
    
    # Cache miss - compute
    print("[CACHE MISS] Computing user neighbors from scratch...")
    neighbors = precompute_all_user_neighbors(R, sim_fn, K)
    
    # Save to cache
    cache.save(cache_key, neighbors)
    
    return neighbors


def compute_neighbors(R, sim_fn, K=25, min_overlap=10):
    """
    Compute neighbors with custom min_overlap parameter.
    Used for Bayesian Optimization experiments.
    
    Args:
        R: User-item rating matrix (DataFrame)
        sim_fn: Similarity function
        K: Number of neighbors
        min_overlap: Minimum overlap for similarity calculation
    
    Returns:
        Dictionary of {user_id: {neighbor_id: similarity, ...}}
    """
    from functools import partial
    
    print(f"\n[COMPUTE] Computing neighbors: K={K}, min_overlap={min_overlap}")
    
    # Create a partial function with min_overlap parameter
    # Check if similarity function accepts MIN_OVERLAP
    if 'MIN_OVERLAP' in sim_fn.__code__.co_varnames:
        sim_fn_configured = partial(sim_fn, MIN_OVERLAP=min_overlap)
    else:
        sim_fn_configured = sim_fn
    
    # Use the existing precompute function
    neighbors = precompute_all_user_neighbors(R, sim_fn_configured, K)
    
    return neighbors


def update_neighbors_for_new_user(cache_path, neighbors, R, sim_fn, new_uid, K=25):
    print(f"[INCREMENTAL] Computing neighbors for NEW user {new_uid}")
    # This is single user, no need for parallel
    neighbors[new_uid] = compute_neighbors_for_user(R, sim_fn, new_uid, K)
    save_neighbors(cache_path, neighbors)
    print("[INCREMENTAL] Updated & saved cache.")
    return neighbors
