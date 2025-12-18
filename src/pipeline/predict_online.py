import os
import sys
import pickle
import json
import time
import numpy as np
import pandas as pd
from typing import List

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))) # Allow 'from recommender...'
from src.recommender.data_loader import load_movies, load_watchlists
from src.recommender.IBCF.item_based_cf import ItemBasedCF
from src.recommender.UBCF.user_based_cf import UserBasedCF
from src.recommender.hybrid.hybrid_model_1 import HybridModel1
from src.recommender.hybrid.hybrid_model_2 import SwitchingHybridRecommender
from src.recommender.hybrid.hybrid_model_3 import WatchlistHybridModel
from src.recommender.hybrid.ensemble_model import EnsembleRecommender

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "model_artifacts")

class OnlineRecommenderService:
    def __init__(self):
        print("[OnlineService] Initializing Service...")
        self.movies = load_movies()
        self.watchlists = load_watchlists()
        self._load_artifacts()
        self._build_pipeline()
        print("[OnlineService] Ready.")
        
    def _load_artifacts(self):
        st = time.time()
        print("  - Loading Artifacts...", end=" ")
        
        with open(os.path.join(ARTIFACTS_DIR, "ensemble_weights.json"), "r") as f:
            self.weights = json.load(f)
            
        with open(os.path.join(ARTIFACTS_DIR, "item_neighbors.pkl"), "rb") as f:
            self.item_neighbors = pickle.load(f)
            
        with open(os.path.join(ARTIFACTS_DIR, "user_neighbors.pkl"), "rb") as f:
            self.user_neighbors = pickle.load(f)
            
        with open(os.path.join(ARTIFACTS_DIR, "full_cf_matrix.pkl"), "rb") as f:
            self.full_cf = pickle.load(f)
            
        with open(os.path.join(ARTIFACTS_DIR, "full_norm_matrix.pkl"), "rb") as f:
            self.full_norm = pickle.load(f)
            
        with open(os.path.join(ARTIFACTS_DIR, "cbf_model.pkl"), "rb") as f:
            self.cbf_model = pickle.load(f)
            
        print(f"Done ({time.time()-st:.2f}s)")
        
    def _build_pipeline(self):
        print("  - Reconstructing Models...", end=" ")
        # Reconstruct Models using Loaded Artifacts
        
        # UBCF
        user_means = self.full_cf.mean(axis=1)
        item_means = self.full_cf.mean(axis=0)
        global_mean = self.full_cf.stack().mean()
        
        self.ibcf = ItemBasedCF(self.full_cf, self.full_norm, self.item_neighbors, self.movies, top_k=20)
        self.ubcf = UserBasedCF(self.full_cf, self.user_neighbors, user_means, item_means, global_mean, self.movies)
        
        # Hybrids
        self.h1 = HybridModel1(self.ibcf, self.cbf_model, C=1.0)
        self.h2 = SwitchingHybridRecommender(self.ubcf, self.cbf_model)
        self.h3 = WatchlistHybridModel(self.movies, self.watchlists, self.cbf_model)
        
        # Ensemble
        self.ensemble = EnsembleRecommender([self.h1, self.h2, self.h3], weights=self.weights)
        print("Done.")

    def get_recommendations(self, group_users, top_k=10, blacklist: List[int]=None):
        """
        Fast Online Prediction.
        """
        st = time.time()
        
        if blacklist is None:
            blacklist = []
        
        # --- 1. Global Popularity (Speed Baseline) ---
        candidates = self.full_cf.count().sort_values(ascending=False).head(300).index.tolist()
        
        # --- 2. Watchlist Integration (Explicit Intent) ---
        wl_items = self.watchlists[self.watchlists['userId'].isin(group_users)]['movieId'].unique().tolist()
        candidates = list(set(candidates + wl_items))
        
        # --- 3. Personalized Retrieval (Collaborative Filtering) ---
        # Retrieve items similar to what users liked (IBCF) or what similar users liked (UBCF)
        personal_candidates = set()
        for uid in group_users:
            if uid in self.full_cf.index:
                # User's recent favorites
                user_hist = self.full_cf.loc[uid].dropna().sort_values(ascending=False).head(5)
                for mid in user_hist.index:
                    if mid in self.item_neighbors:
                        neighbors = list(self.item_neighbors[mid].keys())[:5]
                        personal_candidates.update(neighbors)
                        
                # Similar users' favorites (fixed in previous step)
                if uid in self.user_neighbors:
                    u_neighbors = list(self.user_neighbors[uid].keys())[:5]
                    for neighbor_id in u_neighbors:
                        if neighbor_id in self.full_cf.index:
                            n_top = self.full_cf.loc[neighbor_id].dropna().sort_values(ascending=False).head(3).index.tolist()
                            personal_candidates.update(n_top)
        
        candidates = list(set(candidates + list(personal_candidates)))
        
        # --- 4. Filtering ---
        # Remove movies already watched by the group
        watched_set = set()
        for uid in group_users:
            if uid in self.full_cf.index:
                user_seen = self.full_cf.loc[uid].dropna().index.tolist()
                watched_set.update(user_seen)
        
        watched_set.update(blacklist)
        
        candidates = [c for c in candidates if c not in watched_set]
        
        # Predict
        recs = self.ensemble.recommend(group_users, candidates, top_k=top_k)
        
        et = time.time()
        meta = {
            "latency_ms": (et - st) * 1000,
            "weights_used": self.weights,
            "candidate_count": len(candidates)
        }
        
        return recs, meta

if __name__ == "__main__":
    # Demo Usage
    service = OnlineRecommenderService()
    
    sample_users = [611, 618, 623]
    print(f"\nRequesting Recommendations for Group {sample_users}...")
    
    # Manual Blacklist for Demo (User feedback: 611 has watched 35836)
    manual_blacklist = [35836] 
    recommendations, metadata = service.get_recommendations(sample_users, blacklist=manual_blacklist)
    
    print(f"\nLatency: {metadata['latency_ms']:.2f}ms")
    print("Results:")
    for r in recommendations:
        mid = r['movie_id']
        # Lookup title
        title = service.movies[service.movies['movieId'] == mid]['title'].values[0]
        print(f" - {title} (Score: {r['score']:.2f}) [{r['group_explanation'][:50]}...]")
