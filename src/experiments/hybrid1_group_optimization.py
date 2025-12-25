"""
HYBRID MODEL 1 - GROUP-LEVEL OPTIMIZATION (VECTORIZED)
======================================================
Optimized version of Hybrid1 Group Optimization.
Uses Matrix Operations for fast prediction and accurate NDCG calculation.

RESEARCH QUESTION:
Identify the optimal Trust Factor (C) and Aggregation Strategy.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import time
import math
import scipy.sparse as sp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from recommender.data_loader import load_movies, build_cf_matrix
from recommender.IBCF.neighbors_item import load_or_compute_item_neighbors
from recommender.CB.content_based import ContentBasedModel
from utils.model_utils import normalize_zscore
from evaluation_config import OFFLINE_EVAL_CONFIG
from sklearn.metrics.pairwise import cosine_similarity

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

class FastHybrid1Optimizer:
    def __init__(self, train_df, val_df, movies_df):
        self.train_df = train_df
        self.val_df = val_df 
        self.movies_df = movies_df
        
        print("\n[SETUP] Initializing Fast H1 Components...")
        
        # 1. Build Matrices
        self.cf_matrix = build_cf_matrix(train_df)
        self.norm_matrix = normalize_zscore(self.cf_matrix)
        self.all_movie_ids = list(self.cf_matrix.columns)
        self.movie_to_idx = {mid: i for i, mid in enumerate(self.all_movie_ids)}
        self.num_movies = len(self.all_movie_ids)
        
        # User Stats
        self.user_means = self.cf_matrix.mean(axis=1)
        self.user_stds = self.cf_matrix.std(axis=1).fillna(1.0).replace(0, 1.0)
        
        # 2. IBCF Setup
        print("[SETUP] Loading Item Neighbors...")
        item_sim_matrix = cosine_similarity(self.norm_matrix.fillna(0).T)
        self.item_sim_df = pd.DataFrame(
            item_sim_matrix,
            index=self.norm_matrix.columns,
            columns=self.norm_matrix.columns
        )
        self.item_neighbors = load_or_compute_item_neighbors(
            self.item_sim_df, 
            K=OFFLINE_EVAL_CONFIG['item_k'],
            metric="cosine"
        )
        self._build_sparse_sim_matrix()
        
        # 3. CBF Setup
        self.cbf = ContentBasedModel(movies_df=movies_df, ratings_df=train_df)
        
        print("✅ Optimizer Ready.")

    def _build_sparse_sim_matrix(self):
        rows, cols, data = [], [], []
        for mid, neighbors in self.item_neighbors.items():
            if mid not in self.movie_to_idx: continue
            row_idx = self.movie_to_idx[mid]
            for n_mid, sim in neighbors.items():
                if n_mid in self.movie_to_idx:
                    col_idx = self.movie_to_idx[n_mid]
                    rows.append(row_idx)
                    cols.append(col_idx)
                    data.append(sim)
        
        self.sim_matrix = sp.csr_matrix((data, (rows, cols)), shape=(self.num_movies, self.num_movies))
        self.abs_sim_matrix = sp.csr_matrix((np.abs(data), (rows, cols)), shape=(self.num_movies, self.num_movies))

    def precompute_ibcf_scores(self, group_users, candidates):
        valid_uids = [u for u in group_users if u in self.norm_matrix.index]
        if not valid_uids: return None, None
        
        z_subset = self.norm_matrix.loc[valid_uids].fillna(0).values
        z_sparse = sp.csr_matrix(z_subset)
        
        raw_subset = self.cf_matrix.loc[valid_uids]
        rated_mask = (~raw_subset.isna()).values.astype(float)
        rated_sparse = sp.csr_matrix(rated_mask)
        
        numerators = (z_sparse @ self.sim_matrix.T).toarray()
        denominators = (rated_sparse @ self.abs_sim_matrix.T).toarray()
        
        with np.errstate(divide='ignore', invalid='ignore'):
            pred_z = numerators / denominators
            pred_z[denominators == 0] = np.nan
            
        means = self.user_means.loc[valid_uids].values.reshape(-1, 1)
        stds = self.user_stds.loc[valid_uids].values.reshape(-1, 1)
        pred_ratings = means + (pred_z * stds)
        
        # Support
        adj_data = np.ones_like(self.sim_matrix.data)
        adj_matrix = sp.csr_matrix((adj_data, self.sim_matrix.indices, self.sim_matrix.indptr), shape=self.sim_matrix.shape)
        supports = (rated_sparse @ adj_matrix.T).toarray()
        
        cand_indices = [self.movie_to_idx[m] for m in candidates if m in self.movie_to_idx]
        
        # Return full size arrays for these candidates
        # Shape: [n_users, n_candidates]
        return pred_ratings[:, cand_indices], supports[:, cand_indices]

    def evaluate_group(self, group_users, c_values, agg_strategies=['average']):
        # 1. Ground Truth (Relevant items in validation)
        ground_truth = self.get_group_ground_truth(group_users, strategy='union')
        if not ground_truth: return None
        
        relevant_mids = list(ground_truth.keys())
        
        # 2. Negative Sampling (Standard Standard: Relevant + 100 Random Negatives)
        # Identify watched in training to exclude
        watched_train = set()
        for uid in group_users:
            if uid in self.train_df['userId'].values:
                watched_train.update(self.train_df[self.train_df['userId']==uid]['movieId'].tolist())
        
        # Negatives universe
        all_mids_set = set(self.all_movie_ids)
        available_negatives = list(all_mids_set - watched_train - set(relevant_mids))
        
        rng = np.random.RandomState(42) # Fixed seed for consistency
        if len(available_negatives) > 100:
            negatives = list(rng.choice(available_negatives, size=100, replace=False))
        else:
            negatives = available_negatives
            
        candidates = relevant_mids + negatives
        
        # 3. Predict Raw Scores (Vectorized)
        ibcf_scores, supports = self.precompute_ibcf_scores(group_users, candidates)
        if ibcf_scores is None: return None
        
        # CBF Fallback
        cbf_scores = np.zeros_like(ibcf_scores)
        for i, uid in enumerate(group_users):
            for j, mid in enumerate(candidates):
                 val = self.cbf.predict_rating(uid, mid)
                 cbf_scores[i, j] = val if not np.isnan(val) else 0 # CBF Default
                 
        ibcf_nans = np.isnan(ibcf_scores)
        ibcf_scores[ibcf_nans] = cbf_scores[ibcf_nans]

        results = {}
        
        # Iterate C
        for c in c_values:
            denom = supports + c
            denom[denom == 0] = 1e-9
            alpha = supports / denom
            beta = c / denom
            
            mixed_scores = (alpha * ibcf_scores) + (beta * cbf_scores)
            
            # Iterate Aggregation Strategies
            for agg in agg_strategies:
                if agg == 'average':
                    group_scores = np.nanmean(mixed_scores, axis=0)
                elif agg == 'least_misery':
                    group_scores = np.nanmin(mixed_scores, axis=0)
                elif agg == 'harmonic_mean':
                    safe_scores = np.maximum(mixed_scores, 0.01)
                    inv_sum = np.sum(1.0 / safe_scores, axis=0)
                    group_scores = mixed_scores.shape[0] / inv_sum
                
                # Rank
                top_k_idx = np.argsort(group_scores)[::-1][:10]
                top_ids = [candidates[i] for i in top_k_idx]
                
                ndcg = self.calculate_ndcg(top_ids, ground_truth, 10)
                
                key = (c, agg)
                results[key] = ndcg
            
        return results

    def get_group_ground_truth(self, group_users, strategy="union"):
        # Default robust strategy
        group_val = self.val_df[self.val_df['userId'].isin(group_users)]
        if group_val.empty: return {}
        thresh = OFFLINE_EVAL_CONFIG.get('ground_truth_threshold', 3.5)
        rel = set(group_val[group_val['rating'] >= thresh]['movieId'])
        valid = rel & set(self.all_movie_ids)
        return {mid: 1.0 for mid in valid}

    def calculate_ndcg(self, rec_ids, ground_truth, k):
        rels = [ground_truth.get(mid, 0.0) for mid in rec_ids[:k]]
        dcg = sum([r / math.log2(i+2) for i, r in enumerate(rels)])
        idcg = sum([1.0 / math.log2(i+2) for i in range(min(len(ground_truth), k))])
        return dcg / idcg if idcg > 0 else 0.0

    def create_validation_groups(self, num_groups=30):
        # Filter for active users (Higher threshold)
        counts = self.val_df.groupby('userId').size()
        users = counts[counts >= 5].index.tolist()
        
        if not users:
             print("Warning: No users with >= 5 ratings. Lowering threshold to 3.")
             users = counts[counts >= 3].index.tolist()
             
        rng = np.random.RandomState(42)
        groups = []
        for _ in range(num_groups):
            if len(users) < 3: break
            size = rng.randint(2, 5)
            groups.append(list(rng.choice(users, size=size, replace=False)))
        return groups

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    splits_dir = os.path.join(project_root, "data", "splits")
    
    train_df = pd.read_csv(os.path.join(splits_dir, 'train.csv'))
    val_df = pd.read_csv(os.path.join(splits_dir, 'validation.csv'))
    movies_df = load_movies()
    
    optimizer = FastHybrid1Optimizer(train_df, val_df, movies_df)
    groups = optimizer.create_validation_groups(30)
    
    c_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    strategies = ['average', 'least_misery', 'harmonic_mean']
    
    # Store Sums/Counts
    # Keys: (c, agg)
    agg_sums = {}
    agg_counts = {}
    
    print(f"\nEvaluating {len(groups)} groups on C values {c_values} and Strategies {strategies}...")
    
    for i, g in enumerate(groups):
        print(f"Propcessing Group {i+1}/{len(groups)}...", end='\r')
        res = optimizer.evaluate_group(g, c_values, strategies)
        if res:
            for key, ndcg in res.items():
                if key not in agg_sums:
                    agg_sums[key] = 0.0
                    agg_counts[key] = 0
                agg_sums[key] += ndcg
                agg_counts[key] += 1
                
    print("\n\nEvaluation Complete. Calculating Averages...")
    
    # Format Results
    rows = []
    for c in c_values:
        for agg in strategies:
            key = (c, agg)
            if key in agg_sums:
                avg = agg_sums[key] / agg_counts[key]
                rows.append({'C': c, 'Aggregation': agg, 'NDCG@10': avg})
    
    df = pd.DataFrame(rows)
    print("\nFINAL RESULTS:")
    print(df.sort_values('NDCG@10', ascending=False).to_string(index=False))
    
    out_path = os.path.join(RESULTS_DIR, "hybrid1_optimized_results.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()
