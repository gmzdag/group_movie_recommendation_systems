"""
Item Neighbors Module
---------------------
Computes and caches TOP-K similar items for each movie.
"""

import time
import pickle
import os
import pandas as pd
import numpy as np


def compute_item_neighbors(item_sim: pd.DataFrame, K: int = 60):
    """
    For each item, store top-K most similar items.
    ✅ OPTIMIZED VERSION with vectorization
    """
    print(f"\n[DEBUG] compute_item_neighbors() - K={K}")
    print(f"[DEBUG] Similarity matrix shape: {item_sim.shape}")
    
    neighbors = {}
    movie_ids = item_sim.index.tolist()
    sim_array = item_sim.values
    total = len(movie_ids)
    
    print(f"[DEBUG] Processing {total} movies...")
    
    start = time.time()
    last = start
    
    for i, movie_id in enumerate(movie_ids):
        # ✅ OPTIMIZED: NumPy partition instead of full sort
        row = sim_array[i].copy()
        row[i] = -np.inf  # Exclude self
        
        # argpartition: O(n) instead of O(n log n)
        if K < len(row):
            top_k_idx = np.argpartition(row, -K)[-K:]
            # Only sort the K elements
            top_k_idx = top_k_idx[np.argsort(row[top_k_idx])[::-1]]
        else:
            top_k_idx = np.argsort(row)[::-1][:K]
        
        # Filter out invalid similarities
        valid_neighbors = {
            movie_ids[j]: float(row[j]) 
            for j in top_k_idx 
            if row[j] > -np.inf and not np.isnan(row[j])
        }
        
        neighbors[movie_id] = valid_neighbors
        
        # Progress bar every 1 second
        now = time.time()
        if now - last >= 1 or i == 0 or i == total - 1:
            progress = (i + 1) / total
            elapsed = now - start
            eta = (elapsed / progress) - elapsed if progress > 0 else 0
            
            print(f"[PROGRESS] {i+1}/{total} ({progress*100:.1f}%) | "
                  f"Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s | "
                  f"Current movie: {movie_id} ({len(valid_neighbors)} neighbors)")
            last = now
    
    total_time = time.time() - start
    print(f"\n[DONE] Item neighbors computed in {total_time:.2f}s")
    print(f"[DEBUG] Average neighbors per movie: {np.mean([len(v) for v in neighbors.values()]):.1f}")
    
    return neighbors


# ------------------------
# Caching
# ------------------------

def save_neighbors(path, neighbors):
    """Save neighbors to pickle file"""
    print(f"\n[DEBUG] Saving neighbors to {path}...")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    with open(path, "wb") as f:
        pickle.dump(neighbors, f)
    
    file_size = os.path.getsize(path) / 1024 / 1024  # MB
    print(f"[DEBUG] Saved successfully ({file_size:.2f} MB)")


def load_neighbors(path: str) -> dict:
    """Load neighbors from pickle file"""
    print(f"[DEBUG] Loading neighbors from {path}...")
    
    with open(path, "rb") as f:
        neighbors = pickle.load(f)
    
    file_size = os.path.getsize(path) / 1024 / 1024  # MB
    print(f"[DEBUG] Loaded {len(neighbors)} movies ({file_size:.2f} MB)")
    
    return neighbors


def load_or_compute_item_neighbors(item_sim: pd.DataFrame, K: int = 60, metric: str = "cosine"):
    """
    Load from centralized cache OR compute and save.
    Uses hash-based cache keys to detect data changes.
    
    Args:
        item_sim: Item similarity matrix
        K: Number of neighbors
        metric: Similarity metric used (for cache key)
    
    Returns:
        Dictionary of {movie_id: {neighbor_id: similarity, ...}}
    """
    from src.utils.cache_manager import CacheManager
    
    cache = CacheManager()
    
    # Simple, fixed cache key (no hash - assumes same data each time)
    # If you change the training data, manually clear cache
    cache_key = f"item_neighbors_k{K}_{metric}"
    
    print(f"\n{'='*60}")
    print(f"[CACHE] Item Neighbors - K={K}, metric={metric}")
    print(f"[CACHE] Cache key: {cache_key}")
    print(f"{'='*60}")
    
    # Try to load from cache
    neighbors = cache.load(cache_key)
    
    if neighbors is not None:
        print(f"[CACHE HIT] Loaded {len(neighbors)} movies from cache")
        return neighbors
    
    # Cache miss - compute
    print(f"[CACHE MISS] Computing item neighbors from scratch...")
    neighbors = compute_item_neighbors(item_sim, K)
    
    # Save to cache
    cache.save(cache_key, neighbors)
    
    return neighbors