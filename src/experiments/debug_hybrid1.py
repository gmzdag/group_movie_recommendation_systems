
import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import ndcg_score

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.recommender.data_loader import load_movies, load_train_valid_test_splits, build_cf_matrix
from src.recommender.IBCF.item_based_cf import ItemBasedCF
from src.recommender.IBCF.neighbors_item import load_or_compute_item_neighbors
from src.recommender.CB.content_based import ContentBasedModel
from src.utils.model_utils import normalize_zscore
from sklearn.metrics.pairwise import cosine_similarity

def debug_hybrid1():
    print("--- DEBUGGING HYBRID 1 LOW SCORE ---")
    
    # 1. Load Data
    print("Loading data...")
    train_df, valid_df, _ = load_train_valid_test_splits()
    movies = load_movies()
    
    # 2. Setup IBCF
    print("Setting up IBCF...")
    raw_um = build_cf_matrix(train_df)
    norm_um = normalize_zscore(raw_um)
    
    # similarity
    norm_um_filled = norm_um.fillna(0)
    item_sim = pd.DataFrame(
        cosine_similarity(norm_um_filled.T),
        index=norm_um.columns,
        columns=norm_um.columns
    )
    np.fill_diagonal(item_sim.values, 0)
    
    neighbors = load_or_compute_item_neighbors(item_sim, K=60, metric="cosine_zscore")
    
    ibcf = ItemBasedCF(raw_um, norm_um, neighbors, movies, top_k=60)
    ibcf.global_mean = train_df['rating'].mean()
    
    # 3. Setup CBF
    print("Setting up CBF...")
    cbf = ContentBasedModel(movies, train_df)
    
    # 4. Pick a User from Validation who has good history in Train
    # Find user with overlap
    train_users = set(train_df['userId'].unique())
    valid_users = set(valid_df['userId'].unique())
    common_users = list(train_users & valid_users)
    
    target_user = common_users[0]
    print(f"\nAnalyzing User {target_user}")
    
    # Check History
    train_hist = train_df[train_df['userId'] == target_user]
    print(f"Train History: {len(train_hist)} items")
    
    # Pick a "Ground Truth" item from Validation (Rating >= 4.0)
    valid_hist = valid_df[(valid_df['userId'] == target_user) & (valid_df['rating'] >= 4.0)]
    if valid_hist.empty:
        print("No positive validation items for this user.")
        return

    target_item = valid_hist.iloc[0]['movieId']
    target_rating = valid_hist.iloc[0]['rating']
    print(f"Target Item: {target_item} (Actual Rating: {target_rating})")
    
    # 5. Predict with IBCF
    print("\n[IBCF Trace]")
    ib_pred, info = ibcf.predict(target_user, target_item, return_info=True)
    print(f"IBCF Pred: {ib_pred}")
    print(f"Info: {info}")
    
    if info['n_neighbors'] == 0:
        print(">> IBCF Failed due to 0 neighbors. Checking why...")
        if target_item not in neighbors:
            print(f"   Item {target_item} not in neighbor cache!")
        else:
            item_ns = neighbors[target_item]
            print(f"   Item {target_item} has {len(item_ns)} neighbors in cache.")
            # Check overlap with user history
            hist_ids = train_hist['movieId'].tolist()
            overlap = [nid for nid in item_ns if nid in hist_ids]
            print(f"   Overlap with user history: {len(overlap)}")
            
    # 6. Predict with CBF
    print("\n[CBF Trace]")
    cb_pred = cbf.predict_rating(target_user, target_item)
    print(f"CBF Pred: {cb_pred}")
    
    # 7. Mock Hybrid Score
    C = 1.0
    n = info.get('n_neighbors', 0)
    alpha = n / (n + C)
    beta = C / (n + C)
    
    if np.isnan(ib_pred): ib_pred = 0
    if np.isnan(cb_pred): cb_pred = 0
    
    final = alpha * ib_pred + beta * cb_pred
    print(f"\nHybrid Score (C=1): {final:.4f} (alpha={alpha:.2f}, beta={beta:.2f})")
    
    # 8. Rank Check
    # Compare with 10 random negative items
    print("\n[Ranking Check]")
    all_items = set(raw_um.columns)
    watched = set(train_hist['movieId']) | set(valid_df[valid_df['userId']==target_user]['movieId'])
    candidates = list(all_items - watched)
    np.random.shuffle(candidates)
    negatives = candidates[:10]
    
    items_to_score = [target_item] + negatives
    scores = []
    
    for mid in items_to_score:
        # Mini Hybrid Logic
        p_ib, i = ibcf.predict(target_user, mid, return_info=True)
        p_cb = cbf.predict_rating(target_user, mid)
        
        # Handle nan
        if np.isnan(p_ib) and np.isnan(p_cb): s = 0
        elif np.isnan(p_ib): s = p_cb
        elif np.isnan(p_cb): s = p_ib
        else:
            nn = i.get('n_neighbors', 0)
            al = nn / (nn + C)
            be = C / (nn + C)
            s = al * p_ib + be * p_cb
            
        scores.append((mid, s))
        
    scores.sort(key=lambda x: x[1], reverse=True)
    
    print(f"{'MovieID':<10} | {'Score':<10} | {'Type'}")
    print("-" * 30)
    for mid, s in scores:
        type_ = "TARGET" if mid == target_item else "Noise"
        print(f"{mid:<10} | {s:<10.4f} | {type_}")
        
    rank = [x[0] for x in scores].index(target_item) + 1
    print(f"\nTarget Rank: {rank}/{len(scores)}")

if __name__ == "__main__":
    debug_hybrid1()
