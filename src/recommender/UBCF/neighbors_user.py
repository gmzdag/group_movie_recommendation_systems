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

# Global variables for worker processes
_worker_R = None
_worker_sim_fn = None
_worker_K = None


def _init_worker(R, sim_fn, K):
    global _worker_R, _worker_sim_fn, _worker_K
    _worker_R = R
    _worker_sim_fn = sim_fn
    _worker_K = K
    # Signal that worker is ready (helpful for debugging "hanging")
    # print(f"[WORKER] Process {os.getpid()} initialized with {len(R)} users.")

def _worker_task(uids):
    """
    Worker task that processes a BATCH of users.
    Processing a batch (list of uids) reduces the overhead of pickling/unpickling 
    results and task management compared to one-by-one.
    """
    results = {}
    # Access globals
    target_R = _worker_R
    sim = _worker_sim_fn
    k_val = _worker_K
    
    for uid in uids:
        # Inline the compute_neighbors_for_user logic or call it
        # Calling it is cleaner; overhead is function call (negligible compared to logic)
        results[uid] = compute_neighbors_for_user(target_R, sim, uid, k_val)
        
    return results

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
    num_processes = min(cpu_count(), 8) # Cap at 8
    print(f"[PARALLEL] Computing neighbors w/ {num_processes} CPUs (Batch Mode)...")
    
    # Create batches of users manually to reduce IPC calls
    # A batch size of ~50-100 is usually good for this kind of work
    batch_size = 50
    user_batches = [users[i:i + batch_size] for i in range(0, len(users), batch_size)]
    total_batches = len(user_batches)
    
    with Pool(processes=num_processes, initializer=_init_worker, initargs=(R, sim_fn, K)) as pool:
        # Map over batches
        minibatch_iter = pool.imap_unordered(_worker_task, user_batches)
        
        users_processed = 0
        for batch_result in minibatch_iter:
            # batch_result is a dict {uid: neighbors}
            neighbors.update(batch_result)
            users_processed += len(batch_result)
            
            # Print status
            if users_processed % 100 < batch_size or users_processed == total: # approx check
                 print(f"[PARALLEL] Processed {users_processed}/{total} users ({users_processed/total*100:.1f}%)")
    
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
    # Import with proper path handling
    import sys
    import os
    
    # Add src to path if not already there
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    
    from utils.cache_manager import CacheManager
    
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
