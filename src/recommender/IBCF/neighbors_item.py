"""
Item Neighbors Module
---------------------
Computes and caches TOP-K similar items for each movie.
"""

import time
import pickle
import os
import pandas as pd


def compute_item_neighbors(item_sim: pd.DataFrame, K: int = 60):
    """
    For each item, store top-K most similar items.
    """
    neighbors = {}

    movies = item_sim.index
    total = len(movies)

    start = time.time()
    last = start

    for idx, movie_id in enumerate(movies):
        sims = item_sim.loc[movie_id].drop(movie_id, errors="ignore")
        top_k = sims.sort_values(ascending=False).head(K)

        neighbors[movie_id] = top_k.to_dict()

        # progress bar
        now = time.time()
        if now - last >= 1:
            progress = (idx + 1) / total
            elapsed = now - start
            eta = (elapsed / progress) - elapsed
            print(f"[ITEM-NEIGH] {idx+1}/{total} ({progress*100:.1f}%) | ETA={eta:.1f}s")
            last = now

    print(f"[DONE] Item neighbors computed in {time.time()-start:.1f}s.")
    return neighbors


# ------------------------
# Caching
# ------------------------

def save_neighbors(path: str, obj: dict):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_neighbors(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def load_or_compute_item_neighbors(cache_path: str, item_sim: pd.DataFrame, K: int = 60):
    """
    Load from cache OR compute and save.
    """
    if os.path.exists(cache_path):
        print(f"[CACHE] Loaded item neighbors from {cache_path}")
        return load_neighbors(cache_path)

    print("[CACHE MISS] Computing item neighbors...")
    neigh = compute_item_neighbors(item_sim, K)
    save_neighbors(cache_path, neigh)
    return neigh
