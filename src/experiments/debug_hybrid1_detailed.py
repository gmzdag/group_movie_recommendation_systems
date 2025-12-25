"""
DEBUG HYBRID 1: Detailed Rank Analysis
--------------------------------------
Diagnose why NDCG is low by inspecting the EXACT RANK of relevant items.
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from recommender.data_loader import load_movies, build_cf_matrix
from recommender.IBCF.neighbors_item import load_or_compute_item_neighbors
from recommender.CB.content_based import ContentBasedModel
from utils.model_utils import normalize_zscore
from evaluation_config import OFFLINE_EVAL_CONFIG

def debug_single_user():
    print("="*60)
    print(" DEBUGGING HYBRID 1: SINGLE USER RANK ANALYSIS")
    print("="*60)
    
    # 1. Load Data
    print("Loading data...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    splits_dir = os.path.join(project_root, "data", "splits")
    
    train_df = pd.read_csv(os.path.join(splits_dir, 'train.csv'))
    val_df = pd.read_csv(os.path.join(splits_dir, 'validation.csv'))
    movies_df = load_movies()
    movies_map = dict(zip(movies_df.movieId, movies_df.title))
    
    # 2. Pick Active User
    counts = val_df.groupby('userId').size()
    user_id = counts.idxmax() # User with MOST validation ratings
    
    print(f"\nTarget User: {user_id}")
    print(f"Validation Items (Ground Truth): {counts[user_id]}")
    
    # 3. Ground Truth
    user_val = val_df[val_df['userId'] == user_id]
    liked_items = user_val[user_val['rating'] >= 3.5]['movieId'].tolist()
    print(f"Relevant Items (Rating >= 3.5): {len(liked_items)}")
    for mid in liked_items:
        print(f" - [{mid}] {movies_map.get(mid, 'Unknown')} (Rating: {user_val[user_val['movieId']==mid]['rating'].values[0]})")

    # 4. Prepare Recommendations (Vectorized H1 Logic Sim)
    print("\nPreparing Model Components...")
    cf_matrix = build_cf_matrix(train_df)
    norm_matrix = normalize_zscore(cf_matrix)
    
    # IBCF Signals
    item_sim_matrix = cosine_similarity(norm_matrix.fillna(0).T)
    item_sim_df = pd.DataFrame(item_sim_matrix, index=norm_matrix.columns, columns=norm_matrix.columns)
    neighbors = load_or_compute_item_neighbors(item_sim_df, K=60)
    
    # CBF Signals
    cbf = ContentBasedModel(movies_df, train_df)
    
    # 5. Predict for ALL Candidates
    watched = train_df[train_df['userId'] == user_id]['movieId'].tolist()
    candidates = list(set(cf_matrix.columns) - set(watched))
    
    print(f"\nRanking {len(candidates)} candidates...")
    
    # Manual H1 Prediction Loop (Single User)
    scores = []
    
    # Pre-fetch user data
    if user_id in norm_matrix.index:
        u_vec = norm_matrix.loc[user_id].fillna(0)
    else:
        print("User not in train set!")
        return

    # IBCF Prediction (Vectorized in one line)
    # Score = Sum(Sim * Rating) / Sum(Sim)
    # We do this per candidate
    
    for mid in candidates:
        ib_score = np.nan
        cb_score = np.nan
        support = 0
        
        # IBCF
        if mid in neighbors:
            n_dict = neighbors[mid]
            n_ids = [n for n in n_dict if n in u_vec.index and u_vec[n] != 0]
            if n_ids:
                sims = np.array([n_dict[n] for n in n_ids])
                rats = u_vec[n_ids].values
                ib_score = np.dot(sims, rats) / np.sum(np.abs(sims))
                # Reconstruct
                mean = cf_matrix.loc[user_id].mean()
                std = cf_matrix.loc[user_id].std()
                ib_score = mean + (ib_score * std)
                support = len(n_ids)
        
        # CBF
        cb_score = cbf.predict_rating(user_id, mid)
        
        # Hybrid
        if np.isnan(ib_score): ib_score = cb_score
        if np.isnan(cb_score): cb_score = ib_score
        
        if np.isnan(ib_score) and np.isnan(cb_score):
            final_score = 0
        else:
            C = 1.0
            denom = support + C
            if denom == 0: denom = 1e-9
            alpha = support / denom
            beta = C / denom
            final_score = (alpha * ib_score) + (beta * cb_score)
            
        scores.append((mid, final_score, support))
        
    # 6. Analyze Ranks
    scores.sort(key=lambda x: x[1], reverse=True)
    
    print("\n---------- TOP 10 RECOMMENDATIONS ----------")
    for i, (mid, sc, sup) in enumerate(scores[:10]):
        is_hit = mid in liked_items
        mark = "✅" if is_hit else " "
        print(f"{i+1}. {mark} [{mid}] {movies_map.get(mid, 'Unknown')[:30]:<30} | Score: {sc:.4f} | Sup: {sup}")
        
    print("\n---------- ANALYSIS OF RELEVANT ITEMS ----------")
    hits_found = 0
    for mid in liked_items:
        # Find rank
        try:
            rank = next(i for i, x in enumerate(scores) if x[0] == mid)
            rec_entry = scores[rank]
            print(f"Target [{mid}] {movies_map.get(mid)[:20]:<20} -> Rank: {rank+1}/{len(candidates)} | Score: {rec_entry[1]:.4f} | Sup: {rec_entry[2]}")
            if rank < 10: hits_found += 1
        except StopIteration:
            print(f"Target [{mid}] NOT in candidate list (Filtered?)")
            
    print("\n")
    print(f"NDCG@10 Estimate: {hits_found/min(10, len(liked_items)):.4f}")

if __name__ == "__main__":
    debug_single_user()
