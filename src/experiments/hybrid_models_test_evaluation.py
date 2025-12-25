"""
HYBRID MODELS - FINAL TEST SET EVALUATION
==========================================
Evaluate optimal configurations (found via validation) on the TEST set.

CONFIGURATIONS:
- Hybrid 1: C=1.0, Aggregation=Average
- Hybrid 2: w_ubcf=0.10, w_cbf=0.90, Aggregation=Least Misery

METHODOLOGY:
- 30 test groups (statistical validity)
- Negative Sampling (100 negatives per group)
- Union-based ground truth
- NDCG@10 metric
"""

import os
import sys
import numpy as np
import pandas as pd
import math
import scipy.sparse as sp
from functools import partial

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from recommender.data_loader import load_movies, build_cf_matrix
from recommender.IBCF.neighbors_item import load_or_compute_item_neighbors
from recommender.CB.content_based import ContentBasedModel
from recommender.UBCF.user_based_cf import UserBasedCF
from recommender.UBCF.neighbors_user import load_or_compute_neighbors
from recommender.UBCF.similarity_user import cosine_sim
from utils.model_utils import normalize_zscore
from evaluation_config import OFFLINE_EVAL_CONFIG
from sklearn.metrics.pairwise import cosine_similarity

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================================
# HYBRID 1 EVALUATOR (Vectorized)
# ============================================================================
class Hybrid1TestEvaluator:
    def __init__(self, train_df, test_df, movies_df, C=1.0):
        self.train_df = train_df
        self.test_df = test_df
        self.movies_df = movies_df
        self.C = C
        
        print("\n[H1] Initializing Hybrid 1 (C={})...".format(C))
        
        # Build matrices
        self.cf_matrix = build_cf_matrix(train_df)
        self.norm_matrix = normalize_zscore(self.cf_matrix)
        self.all_movie_ids = list(self.cf_matrix.columns)
        self.movie_to_idx = {mid: i for i, mid in enumerate(self.all_movie_ids)}
        self.num_movies = len(self.all_movie_ids)
        
        self.user_means = self.cf_matrix.mean(axis=1)
        self.user_stds = self.cf_matrix.std(axis=1).fillna(1.0).replace(0, 1.0)
        
        # IBCF
        item_sim_matrix = cosine_similarity(self.norm_matrix.fillna(0).T)
        self.item_sim_df = pd.DataFrame(item_sim_matrix, index=self.norm_matrix.columns, columns=self.norm_matrix.columns)
        self.item_neighbors = load_or_compute_item_neighbors(self.item_sim_df, K=60, metric="cosine")
        self._build_sparse_sim_matrix()
        
        # CBF
        self.cbf = ContentBasedModel(movies_df=movies_df, ratings_df=train_df)
        
        print("[H1] Ready.")
    
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
        
        adj_data = np.ones_like(self.sim_matrix.data)
        adj_matrix = sp.csr_matrix((adj_data, self.sim_matrix.indices, self.sim_matrix.indptr), shape=self.sim_matrix.shape)
        supports = (rated_sparse @ adj_matrix.T).toarray()
        
        cand_indices = [self.movie_to_idx[m] for m in candidates if m in self.movie_to_idx]
        return pred_ratings[:, cand_indices], supports[:, cand_indices]
    
    def evaluate_group(self, group_users):
        ground_truth = self.get_ground_truth(group_users)
        if not ground_truth: return None
        
        relevant_mids = list(ground_truth.keys())
        
        # Negative Sampling
        watched_train = set()
        for uid in group_users:
            if uid in self.train_df['userId'].values:
                watched_train.update(self.train_df[self.train_df['userId']==uid]['movieId'].tolist())
        
        all_mids_set = set(self.all_movie_ids)
        available_negatives = list(all_mids_set - watched_train - set(relevant_mids))
        
        rng = np.random.RandomState(42)
        if len(available_negatives) > 100:
            negatives = list(rng.choice(available_negatives, size=100, replace=False))
        else:
            negatives = available_negatives
        
        candidates = relevant_mids + negatives
        
        # Predict
        ibcf_scores, supports = self.precompute_ibcf_scores(group_users, candidates)
        if ibcf_scores is None: return None
        
        cbf_scores = np.zeros_like(ibcf_scores)
        for i, uid in enumerate(group_users):
            for j, mid in enumerate(candidates):
                val = self.cbf.predict_rating(uid, mid)
                cbf_scores[i, j] = val if not np.isnan(val) else 0
        
        ibcf_nans = np.isnan(ibcf_scores)
        ibcf_scores[ibcf_nans] = cbf_scores[ibcf_nans]
        
        # Hybrid with C
        denom = supports + self.C
        denom[denom == 0] = 1e-9
        alpha = supports / denom
        beta = self.C / denom
        mixed_scores = (alpha * ibcf_scores) + (beta * cbf_scores)
        
        # Average Aggregation
        group_scores = np.nanmean(mixed_scores, axis=0)
        
        top_k_idx = np.argsort(group_scores)[::-1][:10]
        top_ids = [candidates[i] for i in top_k_idx]
        
        return self.calculate_ndcg(top_ids, ground_truth, 10)
    
    def get_ground_truth(self, group_users):
        group_test = self.test_df[self.test_df['userId'].isin(group_users)]
        if group_test.empty: return {}
        thresh = 3.5
        rel = set(group_test[group_test['rating'] >= thresh]['movieId'])
        valid = rel & set(self.all_movie_ids)
        return {mid: 1.0 for mid in valid}
    
    def calculate_ndcg(self, rec_ids, ground_truth, k):
        rels = [ground_truth.get(mid, 0.0) for mid in rec_ids[:k]]
        dcg = sum([r / math.log2(i+2) for i, r in enumerate(rels)])
        idcg = sum([1.0 / math.log2(i+2) for i in range(min(len(ground_truth), k))])
        return dcg / idcg if idcg > 0 else 0.0

# ============================================================================
# HYBRID 2 EVALUATOR
# ============================================================================
class Hybrid2TestEvaluator:
    def __init__(self, train_df, test_df, movies_df, w_ubcf=0.10, w_cbf=0.90, aggregation='average'):
        self.train_df = train_df
        self.test_df = test_df
        self.movies_df = movies_df
        self.w_ubcf = w_ubcf
        self.w_cbf = w_cbf
        self.aggregation = aggregation
        
        print("\n[H2] Initializing Hybrid 2 (w_ubcf={}, w_cbf={}, agg={})...".format(w_ubcf, w_cbf, aggregation))
        
        # Build CF Matrix
        self.cf_matrix = build_cf_matrix(train_df)
        self.all_movie_ids = list(self.cf_matrix.columns)
        
        user_means = self.cf_matrix.mean(axis=1)
        item_means = self.cf_matrix.mean(axis=0)
        global_mean = train_df["rating"].mean()
        
        # UBCF
        neighbors = load_or_compute_neighbors(
            self.cf_matrix,
            partial(cosine_sim, MIN_OVERLAP=5),
            K=50,
            metric="cosine_overlap5_test"
        )
        self.ubcf = UserBasedCF(self.cf_matrix, neighbors, user_means, item_means, global_mean, movies=movies_df)
        
        # CBF
        self.cbf = ContentBasedModel(movies_df, train_df)
        
        print("[H2] Ready.")
    
    def predict(self, uid, mid):
        try:
            ubcf_pred = self.ubcf.predict(uid, mid)
        except:
            ubcf_pred = np.nan
        
        cbf_pred = self.cbf.predict_rating(uid, mid)
        
        if np.isnan(ubcf_pred) and np.isnan(cbf_pred):
            return self.ubcf.global_mean
        elif np.isnan(ubcf_pred):
            return cbf_pred
        elif np.isnan(cbf_pred):
            return ubcf_pred
        
        return self.w_ubcf * ubcf_pred + self.w_cbf * cbf_pred
    
    def evaluate_group(self, group_users):
        ground_truth = self.get_ground_truth(group_users)
        if not ground_truth: return None
        
        relevant_mids = list(ground_truth.keys())
        
        # Negative Sampling
        watched_train = set()
        for uid in group_users:
            if uid in self.train_df['userId'].values:
                watched_train.update(self.train_df[self.train_df['userId']==uid]['movieId'].tolist())
        
        all_mids_set = set(self.all_movie_ids)
        available_negatives = list(all_mids_set - watched_train - set(relevant_mids))
        
        rng = np.random.RandomState(42)
        if len(available_negatives) > 100:
            negatives = list(rng.choice(available_negatives, size=100, replace=False))
        else:
            negatives = available_negatives
        
        candidates = relevant_mids + negatives
        
        # Predict for all candidates
        group_scores = []
        for mid in candidates:
            member_scores = []
            for uid in group_users:
                try:
                    score = self.predict(uid, mid)
                    if not np.isnan(score):
                        member_scores.append(score)
                except:
                    pass
            
            if member_scores:
                # Aggregation Strategy
                if self.aggregation == 'average':
                    final_score = np.mean(member_scores)
                elif self.aggregation == 'least_misery':
                    final_score = np.min(member_scores)
                else:
                    final_score = np.mean(member_scores)  # fallback
                
                group_scores.append((mid, final_score))
        
        if not group_scores: return None
        
        group_scores.sort(key=lambda x: x[1], reverse=True)
        top_ids = [mid for mid, _ in group_scores[:10]]
        
        return self.calculate_ndcg(top_ids, ground_truth, 10)
    
    def get_ground_truth(self, group_users):
        group_test = self.test_df[self.test_df['userId'].isin(group_users)]
        if group_test.empty: return {}
        thresh = 3.5
        rel = set(group_test[group_test['rating'] >= thresh]['movieId'])
        valid = rel & set(self.all_movie_ids)
        return {mid: 1.0 for mid in valid}
    
    def calculate_ndcg(self, rec_ids, ground_truth, k):
        rels = [ground_truth.get(mid, 0.0) for mid in rec_ids[:k]]
        dcg = sum([r / math.log2(i+2) for i, r in enumerate(rels)])
        idcg = sum([1.0 / math.log2(i+2) for i in range(min(len(ground_truth), k))])
        return dcg / idcg if idcg > 0 else 0.0

# ============================================================================
# MAIN EVALUATION
# ============================================================================
def create_test_groups(test_df, num_groups=30):
    counts = test_df.groupby('userId').size()
    users = counts[counts >= 5].index.tolist()
    
    if not users:
        users = counts[counts >= 3].index.tolist()
    
    rng = np.random.RandomState(42)
    groups = []
    for _ in range(num_groups):
        if len(users) < 3: break
        size = rng.randint(2, 5)
        groups.append(list(rng.choice(users, size=size, replace=False)))
    return groups

def main():
    print("="*70)
    print(" HYBRID MODELS - FINAL TEST SET EVALUATION")
    print("="*70)
    
    # Load data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    splits_dir = os.path.join(project_root, "data", "splits")
    
    train_df = pd.read_csv(os.path.join(splits_dir, 'train.csv'))
    test_df = pd.read_csv(os.path.join(splits_dir, 'test.csv'))
    movies_df = load_movies()
    
    print(f"\nTrain: {len(train_df):,} ratings")
    print(f"Test: {len(test_df):,} ratings")
    
    # Create test groups
    groups = create_test_groups(test_df, num_groups=30)
    print(f"\nCreated {len(groups)} test groups")
    
    # Evaluate Hybrid 1
    h1_eval = Hybrid1TestEvaluator(train_df, test_df, movies_df, C=1.0)
    h1_scores = []
    
    print("\n[H1] Evaluating Hybrid 1...")
    for i, g in enumerate(groups):
        print(f"  Group {i+1}/{len(groups)}...", end='\r')
        ndcg = h1_eval.evaluate_group(g)
        if ndcg is not None:
            h1_scores.append(ndcg)
    
    h1_avg = np.mean(h1_scores) if h1_scores else 0.0
    print(f"\n[H1] Hybrid 1 Test NDCG@10: {h1_avg:.4f} (n={len(h1_scores)} groups)")
    
    # Evaluate Hybrid 2 - Average Aggregation
    h2_avg_eval = Hybrid2TestEvaluator(train_df, test_df, movies_df, w_ubcf=0.10, w_cbf=0.90, aggregation='average')
    h2_avg_scores = []
    
    print("\n[H2] Evaluating Hybrid 2 (Average)...")
    for i, g in enumerate(groups):
        print(f"  Group {i+1}/{len(groups)}...", end='\r')
        ndcg = h2_avg_eval.evaluate_group(g)
        if ndcg is not None:
            h2_avg_scores.append(ndcg)
    
    h2_avg_avg = np.mean(h2_avg_scores) if h2_avg_scores else 0.0
    print(f"\n[H2] Hybrid 2 (Average) Test NDCG@10: {h2_avg_avg:.4f} (n={len(h2_avg_scores)} groups)")
    
    # Evaluate Hybrid 2 - Least Misery Aggregation
    h2_lm_eval = Hybrid2TestEvaluator(train_df, test_df, movies_df, w_ubcf=0.10, w_cbf=0.90, aggregation='least_misery')
    h2_lm_scores = []
    
    print("\n[H2] Evaluating Hybrid 2 (Least Misery)...")
    for i, g in enumerate(groups):
        print(f"  Group {i+1}/{len(groups)}...", end='\r')
        ndcg = h2_lm_eval.evaluate_group(g)
        if ndcg is not None:
            h2_lm_scores.append(ndcg)
    
    h2_lm_avg = np.mean(h2_lm_scores) if h2_lm_scores else 0.0
    print(f"\n[H2] Hybrid 2 (Least Misery) Test NDCG@10: {h2_lm_avg:.4f} (n={len(h2_lm_scores)} groups)")
    
    # Final Report
    print("\n" + "="*70)
    print(" FINAL TEST RESULTS")
    print("="*70)
    print(f"\nHybrid 1 (C=1.0, Average):              NDCG@10 = {h1_avg:.4f}")
    print(f"Hybrid 2 (w=0.10/0.90, Average):        NDCG@10 = {h2_avg_avg:.4f}")
    print(f"Hybrid 2 (w=0.10/0.90, Least Misery):   NDCG@10 = {h2_lm_avg:.4f}")
    
    print("\n" + "-"*70)
    if h1_avg > 0 and h2_avg_avg > 0:
        improvement_avg = ((h1_avg / h2_avg_avg) - 1) * 100
        print(f"H1 vs H2(Average):       {improvement_avg:+.1f}%")
    
    if h1_avg > 0 and h2_lm_avg > 0:
        improvement_lm = ((h1_avg / h2_lm_avg) - 1) * 100
        print(f"H1 vs H2(Least Misery):  {improvement_lm:+.1f}%")
    
    if h2_avg_avg > 0 and h2_lm_avg > 0:
        h2_diff = ((h2_lm_avg / h2_avg_avg) - 1) * 100
        print(f"H2: LM vs Average:       {h2_diff:+.1f}%")
    
    # Save results
    results = pd.DataFrame([
        {'Model': 'Hybrid 1', 'Config': 'C=1.0, Average', 'Test_NDCG@10': h1_avg, 'Num_Groups': len(h1_scores)},
        {'Model': 'Hybrid 2', 'Config': 'w=0.10/0.90, Average', 'Test_NDCG@10': h2_avg_avg, 'Num_Groups': len(h2_avg_scores)},
        {'Model': 'Hybrid 2', 'Config': 'w=0.10/0.90, Least Misery', 'Test_NDCG@10': h2_lm_avg, 'Num_Groups': len(h2_lm_scores)}
    ])
    
    out_path = os.path.join(RESULTS_DIR, "hybrid_models_test_results.csv")
    results.to_csv(out_path, index=False)
    print(f"\n💾 Saved to {out_path}")
    print("="*70)

if __name__ == "__main__":
    main()
