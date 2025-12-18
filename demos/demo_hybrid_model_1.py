
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from src.recommender.data_loader import load_ratings, load_movies, build_cf_matrix
from src.recommender.IBCF.item_based_cf import ItemBasedCF
from src.recommender.CB.content_based import ContentBasedModel
from src.recommender.IBCF.neighbors_item import load_or_compute_item_neighbors
from src.recommender.hybrid.hybrid_model_1 import HybridModel1

def normalize_zscore(mat):
    print("Normalizing ratings (Z-Score)...")
    mean = mat.mean(axis=1)
    std = mat.std(axis=1).replace(0, 1)
    return mat.sub(mean, axis=0).div(std, axis=0).fillna(0)

def main():
    print("=== Hybrid Model 1 Demo (Group Recommendation) ===\n")
    
    # 1. Load Data
    print("[1/3] Loading Data & Initializing Models...")
    movies = load_movies()
    ratings = load_ratings()
    
    # Build core matrices
    raw_um = build_cf_matrix(ratings)
    norm_um = normalize_zscore(raw_um)
    
    # Load Neighbors (Assumes cache exists from previous steps for speed)
    cache_path = os.path.join("data", "cache", "item_neighbors.pkl")
    if not os.path.exists(cache_path):
        print("Error: Neighbors cache not found. Please run experiments or setup first.")
        # Fallback handling could go here, but keeping demo simple
        return

    item_neighbors = load_or_compute_item_neighbors(cache_path, None, K=20)
    
    # Init Models
    ib_model = ItemBasedCF(raw_um, norm_um, item_neighbors, movies, top_k=20)
    cb_model = ContentBasedModel(movies, ratings)
    
    # Start Hybrid Model (C=1 based on experiments)
    hybrid = HybridModel1(ib_model, cb_model, C=1.0)
    
    # 2. Define Scenario
    group_users = [611, 618, 623]
    print(f"\n[2/3] Selected Group: {group_users}")
    
    # candidates: top 200 popular movies
    candidates = ratings['movieId'].value_counts().head(200).index.tolist()
    
    # 3. Validation
    print("[3/3] Running 'recommend_for_group'...")
    recommendations = hybrid.recommend_for_group(group_users, candidates, top_k=10)
    
    print("\n=== FINAL RECOMMENDATIONS ===\n")
    
    for rank, rec in enumerate(recommendations, 1):
        mid = rec['movie_id']
        title = movies[movies['movieId'] == mid]['title'].values[0]
        score = rec['score']
        group_expl = rec.get('group_explanation', "")
        
        print(f"#{rank} {title}")
        print(f"   Score: {score:.2f}")
        print(f"   GROUP INSIGHT: {group_expl}")
        print("   Individual Reasons:")
        
        # Display Explanations
        for uid, expl_obj in rec['explanations'].items():
            reason = expl_obj['final_reason']
            print(f"    - User {uid}: {reason}")
        print("-" * 50)
        
    print("\nDemo Completed Successfully.")

if __name__ == "__main__":
    main()
