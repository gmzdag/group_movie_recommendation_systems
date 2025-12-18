
import os
import sys
import numpy as np
import pandas as pd
import warnings

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))) # Add src to path for 'from recommender...' imports

from src.recommender.data_loader import load_movies, load_ratings, load_watchlists, build_cf_matrix
from src.recommender.data_splitter import temporal_train_validation_test_split
from src.experiments.evaluate_helpers import calculate_ndcg

# Models
from src.recommender.IBCF.item_based_cf import ItemBasedCF
from src.recommender.IBCF.neighbors_item import compute_item_neighbors
from src.recommender.UBCF.user_based_cf import UserBasedCF
from src.recommender.UBCF.neighbors_user import precompute_all_user_neighbors
from src.recommender.UBCF.similarity_user import pearson_shrink
from src.recommender.CB.content_based import ContentBasedModel

# Hybrids
from src.recommender.hybrid.hybrid_model_1 import HybridModel1
from src.recommender.hybrid.hybrid_model_2 import SwitchingHybridRecommender
from src.recommender.hybrid.hybrid_model_3 import WatchlistHybridModel
from src.recommender.hybrid.ensemble_model import EnsembleRecommender

from sklearn.metrics.pairwise import cosine_similarity

def normalize_zscore(mat):
    mean = mat.mean(axis=1)
    std = mat.std(axis=1).replace(0, 1)
    return mat.sub(mean, axis=0).div(std, axis=0).fillna(0)

def main():
    warnings.filterwarnings('ignore')
    print("=== Ensemble Recommender Evaluation ===")
    
    # 1. Load Data
    print("[1] Loading Data...")
    movies = load_movies()
    ratings = load_ratings()
    watchlists = load_watchlists()

    # Subsample for fast demo
    print("[WARN] Subsampling data to 10,000 ratings for speed...")
    ratings = ratings.sort_values('timestamp').tail(10000) # Keep recent 10k items
    
    # 2. Split Data (Time-Based)
    print("[2] Splitting Data (Time-Based)...")
    train, val, test = temporal_train_validation_test_split(ratings, train_ratio=0.7, valid_ratio=0.15)
    print(f"    Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    
    # 3. Build Matrices for Training
    print("[3] Building Training Matrices...")
    train_matrix_cf = build_cf_matrix(train) # NaN
    train_matrix_norm = normalize_zscore(train_matrix_cf)
    
    # 4. Initialize Base Models
    print("[4] Initializing Base Models...")
    
    # --- IBCF ---
    print("    - ItemBasedCF: Computing Neighbors...")
    item_sim = cosine_similarity(train_matrix_norm.fillna(0).T)
    item_sim_df = pd.DataFrame(item_sim, index=train_matrix_norm.columns, columns=train_matrix_norm.columns)
    item_neighbors = compute_item_neighbors(item_sim_df, K=20)
    
    ibcf = ItemBasedCF(train_matrix_cf, train_matrix_norm, item_neighbors, movies, top_k=20)
    
    # --- UBCF ---
    print("    - UserBasedCF: Computing Neighbors...")
    user_neighbors = precompute_all_user_neighbors(train_matrix_cf, pearson_shrink, K=50)
    
    user_means = train_matrix_cf.mean(axis=1)
    item_means = train_matrix_cf.mean(axis=0)
    global_mean = train['rating'].mean()
    
    ubcf = UserBasedCF(train_matrix_cf, user_neighbors, user_means, item_means, global_mean, movies=movies)
    
    # --- CBF ---
    print("    - ContentBasedModel...")
    cbf = ContentBasedModel(movies, train) # Train ratings for profile building
    
    # 5. Initialize Hybrid Models
    print("[5] Initializing Hybrid Models...")
    
    # Model 1: Dynamic Trust (IB + CB)
    m1 = HybridModel1(ibcf, cbf, C=1.0)
    
    # Model 2: Switching (UBCF -> CBF)
    m2 = SwitchingHybridRecommender(ubcf, cbf)
    
    # Model 3: Watchlist Matches
    m3 = WatchlistHybridModel(movies, watchlists, cbf)
    
    # 6. Calculate Weights (NDCG on Validation)
    print("[6] Calculating Model Weights (NDCG on Validation Set)...")
    
    # For speed, compute on a sample of validation users?
    val_sample = val.sample(frac=0.5, random_state=42) if len(val) > 2000 else val
    print(f"    Evaluated on {len(val_sample)} interactions...")
    
    ndcg_1 = calculate_ndcg(m1, val_sample)
    print(f"    - Model 1 (Dynamic IB/CB): {ndcg_1:.4f}")
    
    ndcg_2 = calculate_ndcg(m2, val_sample)
    print(f"    - Model 2 (Switching UB/CB): {ndcg_2:.4f}")
    
    ndcg_3 = calculate_ndcg(m3, val_sample)
    print(f"    - Model 3 (Watchlist):     {ndcg_3:.4f}")
    
    # Weight Calculation
    total_score = ndcg_1 + ndcg_2 + ndcg_3
    if total_score > 0:
        w1 = ndcg_1 / total_score
        w2 = ndcg_2 / total_score
        w3 = ndcg_3 / total_score
    else:
        w1, w2, w3 = 0.33, 0.33, 0.33
        
    print(f"\n[WEIGHTS] M1: {w1:.2f}, M2: {w2:.2f}, M3: {w3:.2f}")
    
    # 7. Create Ensemble
    print("[7] Creating Ensemble Recommender...")
    ensemble = EnsembleRecommender(models=[m1, m2, m3], weights=[w1, w2, w3])
    
    # 8. Demo Prediction on Test Group
    print("[8] Running Ensemble Prediction (Demo)...")
    
    # Create a synthetic group from Test users
    test_users = test['userId'].unique()
    if len(test_users) >= 3:
        group_demo = list(test_users[:3])
    else:
        group_demo = [1, 2, 3] # Fallback
        
    print(f"    Group: {group_demo}")
    
    # Candidates: Popular movies not seen by group?
    # Simple candidates: top 500 movies from train set
    candidates = train['movieId'].value_counts().head(500).index.tolist()
    
    # Filter seen items (Simulating filter logic)
    watched_in_train = set()
    for uid in group_demo:
        user_seen = train[train['userId'] == uid]['movieId'].tolist()
        watched_in_train.update(user_seen)
        
    candidates = [c for c in candidates if c not in watched_in_train]
    
    # Run recommend
    recs = ensemble.recommend(group_demo, candidates, top_k=5)
    
    print("\n=== Ensemble Output ===")
    for i, rec in enumerate(recs, 1):
        mid = rec['movie_id']
        title = movies[movies['movieId'] == mid]['title'].values[0] if not movies[movies['movieId'] == mid].empty else str(mid)
        score = rec['score']
        reason = rec['group_explanation']
        src_idx = rec.get('source_model_idx', -1)
        src_name = ["Hybrid 1", "Hybrid 2", "Hybrid 3"][src_idx] if 0 <= src_idx <= 2 else "Unknown"
        
        print(f"#{i} {title}")
        print(f"   Score: {score:.4f}")
        print(f"   Dominant Model: {src_name}")
        print(f"   Reason: {reason}")
        print("-" * 30)

if __name__ == "__main__":
    main()
