import os
import sys
import pickle
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))) # Allow 'from recommender...'

from src.recommender.data_loader import load_movies, load_ratings, load_watchlists, build_cf_matrix
from src.recommender.data_splitter import temporal_train_validation_test_split
from src.experiments.evaluate_helpers import calculate_ndcg
from src.recommender.IBCF.neighbors_item import compute_item_neighbors
from src.recommender.UBCF.neighbors_user import precompute_all_user_neighbors
from src.recommender.UBCF.similarity_user import pearson_shrink
from src.recommender.CB.content_based import ContentBasedModel
from src.recommender.IBCF.item_based_cf import ItemBasedCF
from src.recommender.UBCF.user_based_cf import UserBasedCF
from src.recommender.hybrid.hybrid_model_1 import HybridModel1
from src.recommender.hybrid.hybrid_model_2 import SwitchingHybridRecommender
from src.recommender.hybrid.hybrid_model_3 import WatchlistHybridModel

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "model_artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

def train_and_save():
    warnings.filterwarnings('ignore')
    print("=== STARTING OFFLINE TRAINING ===")
    
    # 1. Load FULL Data
    print("[1] Loading Full Data...")
    movies = load_movies()
    ratings = load_ratings()
    watchlists = load_watchlists()
    
    # 2. Split for Weight Tuning (Validation) 
    # NOTE: Even though we train on strict history, we need weights.
    # We use the Time-Based Split to calculate weights on the Validation set.
    # AFTER calculating weights, we should ideally retrain on Full Data (Train+Val) 
    # to maximize knowledge for the Online Phase.
    
    print("[2] Splitting for Hyperparameter Tuning (using subsample for speed)...")
    # Subsample for tuning to avoid slow full matrix operations just for weights
    # 20% of data is usually enough to find relative model performance
    ratings_tuning_sample = ratings.sample(frac=0.3, random_state=42)
    train_tuning, val_tuning, _ = temporal_train_validation_test_split(ratings_tuning_sample, train_ratio=0.7, valid_ratio=0.15)
    
    # --- STEP A: Calculate Weights using Tuning Set ---
    print("[STEP A] Calculating Ensemble Weights...")
    
    # Build models on Tuning Train
    tt_cf = build_cf_matrix(train_tuning)
    tt_norm = (tt_cf - tt_cf.mean(axis=1).values[:, None]).fillna(0)
    
    # IBCF Tuning
    item_sim_tuning = cosine_similarity(tt_norm.T)
    item_sim_df_tuning = pd.DataFrame(item_sim_tuning, index=tt_norm.columns, columns=tt_norm.columns)
    neighbors_item_tuning = compute_item_neighbors(item_sim_df_tuning, K=20)
    m1_ibcf = ItemBasedCF(tt_cf, tt_norm, neighbors_item_tuning, movies, top_k=20)
    
    # UBCF Tuning
    # Fast neighbors for tuning
    print("    - Computing Tuning Neighbors (UBCF)...")
    from src.recommender.UBCF.similarity_user import cosine_sim # Faster than pearson for weight tuning check
    neighbors_user_tuning = precompute_all_user_neighbors(tt_cf, cosine_sim, K=50) 
    m2_ubcf = UserBasedCF(tt_cf, neighbors_user_tuning, tt_cf.mean(axis=1), tt_cf.mean(axis=0), train_tuning['rating'].mean(), movies)
    
    # CBF Tuning
    m_cbf = ContentBasedModel(movies, train_tuning)
    
    # Hybrids Tuning
    h1 = HybridModel1(m1_ibcf, m_cbf, C=1.0)
    h2 = SwitchingHybridRecommender(m2_ubcf, m_cbf)
    h3 = WatchlistHybridModel(movies, watchlists, m_cbf)
    
    # NDCG
    print("    - Validating Models...")
    # Sample validation for speed
    val_sample = val_tuning.sample(frac=0.2, random_state=42) if len(val_tuning) > 1000 else val_tuning
    
    s1 = calculate_ndcg(h1, val_sample)
    s2 = calculate_ndcg(h2, val_sample)
    s3 = calculate_ndcg(h3, val_sample)
    
    total = s1 + s2 + s3
    if total == 0: weights = [0.33, 0.33, 0.33]
    else: weights = [s1/total, s2/total, s3/total]
    
    print(f"    - Tuning Results: M1={s1:.3f}, M2={s2:.3f}, M3={s3:.3f}")
    print(f"    - Final Weights: {weights}")
    
    # --- STEP B: Train Final Models on FULL AVAILABLE DATA ---
    # User Requirement: "kullanıcıların izlediği tüm filmleri train olarak kullanmayı unutma"
    # So we use `ratings` (Full) to build final matrices.
    
    print("[STEP B] Training Final Matrices on FULL History...")
    
    full_cf = build_cf_matrix(ratings)
    full_norm = (full_cf - full_cf.mean(axis=1).values[:, None]).fillna(0)
    
    # 1. Item Neighbors (Slowest part usually)
    print("    - Computing Full Item Similarity...")
    # Note: filling 0 assumes mean centering handled by full_norm
    full_item_sim = cosine_similarity(full_norm.T)
    full_item_sim_df = pd.DataFrame(full_item_sim, index=full_norm.columns, columns=full_norm.columns)
    final_item_neighbors = compute_item_neighbors(full_item_sim_df, K=20)
    
    # 2. User Neighbors
    print("    - Computing Full User Neighbors...")
    final_user_neighbors = precompute_all_user_neighbors(full_cf, pearson_shrink, K=50)
    
    # 3. Content Vectors
    print("[3] Saving Artifacts...")
    
    # Weights
    with open(os.path.join(ARTIFACTS_DIR, "ensemble_weights.json"), "w") as f:
        json.dump(weights, f)
        
    # Item Neighbors
    with open(os.path.join(ARTIFACTS_DIR, "item_neighbors.pkl"), "wb") as f:
        pickle.dump(final_item_neighbors, f)
        
    # User Neighbors
    with open(os.path.join(ARTIFACTS_DIR, "user_neighbors.pkl"), "wb") as f:
        pickle.dump(final_user_neighbors, f)
        
    # Full Matrices (needed for prediction lookups)
    with open(os.path.join(ARTIFACTS_DIR, "full_cf_matrix.pkl"), "wb") as f:
        pickle.dump(full_cf, f)

    with open(os.path.join(ARTIFACTS_DIR, "full_norm_matrix.pkl"), "wb") as f:
        pickle.dump(full_norm, f)
        
    # CBF Model
    print("    - Training Final CBF Model...")
    final_cbf = ContentBasedModel(movies, ratings)
    with open(os.path.join(ARTIFACTS_DIR, "cbf_model.pkl"), "wb") as f:
        pickle.dump(final_cbf, f)
        
    print("=== OFFLINE TRAINING COMPLETE ===")
    print(f"Artifacts saved to: {ARTIFACTS_DIR}")

if __name__ == "__main__":
    train_and_save()
