
import sys
import os
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from recommender.data_loader import load_all_data
from recommender.UBCF.user_based_cf import UserBasedCF
from recommender.UBCF.neighbors_user import load_or_compute_neighbors
from recommender.UBCF.similarity_user import pearson_shrink
from recommender.CB.content_based import ContentBasedModel
from recommender.hybrid.hybrid_model_2 import SwitchingHybridRecommender

# Cache paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
CACHE_SHR = os.path.join(CACHE_DIR, "user_neighbors_shrink_fixed.pkl") 

def main():
    print("=== Hybrid Model 2 (Switching) - Group Recommendation Demo ===")
    
    # 1. Load Data
    print("\n[1] Loading Data...")
    movies, ratings, watchlists, R_cf, R_dense = load_all_data()
    
    # 2. Init UBCF
    print("[2] Initializing UBCF...")
    global_mean = ratings["rating"].mean()
    user_means = R_cf.mean(axis=1)
    item_means = R_cf.mean(axis=0)
    
    # Compute neighbors
    neighbors = load_or_compute_neighbors(CACHE_SHR, R_cf, pearson_shrink)
    
    ubcf = UserBasedCF(R_cf, neighbors, user_means, item_means, global_mean, movies=movies)
    
    # 3. Init CBF
    print("[3] Initializing CBF...")
    cbf = ContentBasedModel(movies, ratings)
    
    # 4. Init Hybrid
    print("[4] Initializing Switching Hybrid...")
    hybrid = SwitchingHybridRecommender(ubcf, cbf)
    
    # 5. Group Recommendation
    group_users = [618, 623] # Standard test group
    print(f"\n[5] Generating Group Recommendations for Users: {group_users}")
    print("    Strategy: Mean Aggregation of Switching Hybrid Scores")
    
    top_n = 10
    group_recs_df = hybrid.recommend_group(group_users, top_n=top_n)
    
    print("\n=== FINAL RECOMMENDATIONS ===\n")
    
    if group_recs_df.empty:
        print("No recommendations found.")
    else:
        for idx, row in group_recs_df.iterrows():
            rank = idx + 1
            mid = row['movieId']
            score = row['score']
            title = row.get('title', f"Movie {mid}")
            
            print(f"#{rank} {title}")
            print(f"   Score: {score:.4f}")
            # Note: Detailed explanations are not returned by current recommend_group in hybrid_model_2
            print("-" * 30)

    print("\nDemo Completed.")

if __name__ == "__main__":
    main()
