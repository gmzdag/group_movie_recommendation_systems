
import os
import sys
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.recommender.data_loader import load_movies, load_ratings, build_cf_matrix
from src.recommender.IBCF.item_based_cf import ItemBasedCF
from src.recommender.IBCF.neighbors_item import compute_item_neighbors
from src.recommender.CB.content_based import ContentBasedModel
from src.recommender.hybrid.hybrid_model_1 import HybridModel1

from sklearn.metrics.pairwise import cosine_similarity

def normalize_zscore(mat):
    mean = mat.mean(axis=1)
    std = mat.std(axis=1).replace(0, 1)
    return mat.sub(mean, axis=0).div(std, axis=0).fillna(0)

def main():
    print("=== Demo: Hybrid Model 1 (Dynamic Trust) ===")
    
    # 1. Load Data
    print("Loading Data...")
    movies = load_movies()
    ratings = load_ratings()
    
    # Subsample for speed
    ratings = ratings.head(50000)
    
    # 2. Build Components
    print("Building Components...")
    R_cf = build_cf_matrix(ratings)
    R_norm = normalize_zscore(R_cf)
    
    print("Computing Neighbors...")
    sim = cosine_similarity(R_norm.fillna(0).T)
    sim_df = pd.DataFrame(sim, index=R_norm.columns, columns=R_norm.columns)
    neighbors = compute_item_neighbors(sim_df, K=20)
    
    # 3. Models
    print("Initializing Models...")
    ib_model = ItemBasedCF(R_cf, R_norm, neighbors, movies, top_k=20)
    cb_model = ContentBasedModel(movies, ratings)
    
    hybrid = HybridModel1(ib_model, cb_model, C=1.0)
    
    # 4. Predict
    user_id = ratings['userId'].iloc[0]
    movie_id = ratings['movieId'].iloc[0] # Some movie they saw
    
    print(f"\nUser: {user_id}, Movie: {movie_id}")
    
    score = hybrid.predict(user_id, movie_id)
    print(f"Hybrid Score: {score:.2f}")
    
    # 5. Explain
    print("\nExplanations:")
    expl = hybrid.explain(user_id, movie_id)
    print(expl)

if __name__ == "__main__":
    main()
