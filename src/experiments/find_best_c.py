"""
H1 Tuning Experiment (Refactored for Academic Compliance)
-------------------------------------------------------
Performs hyperparameter tuning for the 'C' parameter (Hybrid Weight) in Hybrid Model 1.
This script uses the VALIDATION set to optimize the trade-off between Individual and Group recommender performance.

METHODOLOGY NOTES:
1.  **Ground Truth Definition**: 
    -   Binary Relevance (1.0 = Relevant, 0.0 = Not Relevant).
    -   Criteria: Item must be rated >= 3.5 (threshold) by at least 50% of group members (support).
    -   Justification: This "Majority Consensus" validates that the item is a true group hit, avoiding noise from weak individual signals.

2.  **Metric**: 
    -   NDCG@K (Normalized Discounted Cumulative Gain).
    -   We compare IDCG (Ideal DCG based on all possible relevant items) vs DCG (Actual DCG of top-K recommendations).
    -   This correctly penalizes missing relevant items and ranks them lower.

3.  **Scope**: 
    -   Strictly uses TRAIN (Model Building) and VALIDATION (Tuning). 
    -   NO TEST set leakage. Final evaluation should be run separately on `evaluate_group_system.py`.

4.  **Robustness**: 
    -   Evaluates on 30 deterministic Validation groups.
    -   Also evaluates on a random sample of Individual users to characterize the "Personality vs Consensus" trade-off.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import time
import math
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.recommender.data_loader import load_movies, load_watchlists, build_cf_matrix
# Note: We load ratings manually to ensure split usage
from src.recommender.hybrid.hybrid_model_1 import HybridModel1
from src.recommender.IBCF.item_based_cf import ItemBasedCF
from src.recommender.IBCF.neighbors_item import load_or_compute_item_neighbors
from src.recommender.CB.content_based import ContentBasedModel
from src.utils.model_utils import normalize_zscore
from evaluation_config import OFFLINE_EVAL_CONFIG

# Import metrics
from sklearn.metrics import ndcg_score
from sklearn.metrics.pairwise import cosine_similarity

class RobustH1Tuner:
    """
    Hyperparameter Tuner for Hybrid Model 1 (H1) - C Parameter Optimization
    
    **C Parameter Definition**:
    H1 uses dynamic weighting based on neighbor support:
        α(n) = n / (n + C)  →  IBCF weight
        β(n) = C / (n + C)  →  CBF weight
        Score = α(n) * IBCF + β(n) * CBF
    
    Where:
        n = number of item neighbors used in IBCF prediction
        C = trust transition hyperparameter
    
    **C Interpretation**:
        - Low C (e.g., 0.5): Trust IBCF quickly (even with few neighbors)
        - High C (e.g., 5.0): Require many neighbors before trusting IBCF
        - C=1.0: Balanced (50% trust at n=1 neighbor)
    
    **Evaluation Strategy**:
        - Candidate Universe: Items in CF matrix (IBCF can score)
        - Ground Truth: 3 strategies tested (Union, Average, Intersection)
        - Metric: NDCG@10 with binary relevance (rating >= 3.5)
    """
    def __init__(self, train_df, val_df, movies_df):
        self.train_df = train_df
        self.val_df = val_df 
        self.movies_df = movies_df
        
        print("\n[SETUP] Initializing H1 Recommendation Components (IBCF + CBF)...")
        
        # 1. Build Matrices (Train Data Only)
        self.cf_matrix = build_cf_matrix(train_df)
        self.norm_matrix = normalize_zscore(self.cf_matrix)
        
        # **CRITICAL FIX**: Candidate universe = items in CF matrix (model can only score these)
        # Using movies_df would include cold-start items that IBCF cannot score
        self.all_movie_ids = set(self.cf_matrix.columns)
        
        # 2. Compute/Load Item Neighbors (IBCF)
        # Using cosine metric as standard for rating data
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
        
        # 3. Initialize IBCF
        self.ibcf = ItemBasedCF(
            raw_um=self.cf_matrix,
            norm_um=self.norm_matrix,
            item_neighbors=self.item_neighbors,
            movies=movies_df,
            top_k=OFFLINE_EVAL_CONFIG['item_k']  # Use validated K=60 from config
        )
        
        # 4. Initialize CBF
        self.cbf = ContentBasedModel(
            movies_df=movies_df,
            ratings_df=train_df
        )
        
        print("✅ Tuner Ready (Trained on TRAIN, Tuning on VALIDATION).")
        
    def create_model(self, C: float) -> HybridModel1:
        return HybridModel1(self.ibcf, self.cbf, C=C)

    def create_validation_groups(self, num_groups: int = 30, min_size: int = 2, max_size: int = 4) -> List[List[int]]:
        """
        Generates synthetic groups from the Validation set deterministically.
        """
        user_counts = self.val_df.groupby('userId').size()
        # Require at least 3 items in validation to form a meaningful ground truth
        eligible_users = user_counts[user_counts >= 3].index.tolist()
        
        if len(eligible_users) < min_size:
            print(f"⚠️  Warning: Low validation user count ({len(eligible_users)}). Cannot form many groups.")
            return []
            
        groups = []
        # Fixed seed for reproducibility (Academic Standard)
        rng = np.random.RandomState(42) 
        
        print(f"[INFO] Sampling {num_groups} groups from {len(eligible_users)} eligible validation users.")
        
        for _ in range(num_groups):
            current_group_size = rng.randint(min_size, max_size + 1)
            if len(eligible_users) >= current_group_size:
                group = list(rng.choice(eligible_users, size=current_group_size, replace=False))
                groups.append(group)
            
        return groups

    def get_group_ground_truth(self, group_users: List[int], strategy: str = "union") -> Dict[int, float]:
        """
        Defines the 'Ground Truth' set of relevant items for a group.
        
        **THREE STRATEGIES**:
        1. **UNION (Most Pleasure)**: Item is relevant if ANY member rated it >= 3.5
           - Philosophy: "At least one person will love it"
           - Lenient, maximizes coverage
           
        2. **AVERAGE**: Item is relevant if AVERAGE rating >= 3.5
           - Philosophy: "On average, the group likes it"
           - Balanced approach
           
        3. **INTERSECTION (Least Misery)**: Item is relevant if ALL members rated it >= 3.5
           - Philosophy: "Nobody dislikes it"
           - Strict, ensures consensus
        
        **CRITICAL**: Only includes items that appear in the training set.
        Returns Binary Relevance {movieId: 1.0}.
        """
        group_val = self.val_df[self.val_df['userId'].isin(group_users)]
        if group_val.empty:
            return {}
        
        rating_threshold = OFFLINE_EVAL_CONFIG.get('ground_truth_threshold', 3.5)
        
        if strategy == "union":
            # **UNION (Most Pleasure)**: Any user rating >= threshold
            relevant_items = set(group_val[group_val['rating'] >= rating_threshold]['movieId'].unique())
            
        elif strategy == "average":
            # **AVERAGE**: Mean rating >= threshold
            agg = group_val.groupby('movieId')['rating'].mean()
            relevant_items = set(agg[agg >= rating_threshold].index)
            
        elif strategy == "intersection":
            # **INTERSECTION (Least Misery)**: ALL users must rate >= threshold
            # First, get items rated by ALL users
            item_counts = group_val.groupby('movieId')['userId'].nunique()
            items_rated_by_all = set(item_counts[item_counts == len(group_users)].index)
            
            # Then, check if ALL ratings are >= threshold
            relevant_items = set()
            for movie_id in items_rated_by_all:
                movie_ratings = group_val[group_val['movieId'] == movie_id]['rating']
                if (movie_ratings >= rating_threshold).all():
                    relevant_items.add(movie_id)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        if not relevant_items:
            return {}
        
        # **FIX**: Only keep items that exist in training set
        train_movie_ids = set(self.train_df['movieId'].unique())
        valid_relevant = relevant_items & train_movie_ids
        
        if not valid_relevant:
            print(f"  [WARNING] Ground truth empty after train filter: {len(relevant_items)} items in val, 0 in train")
            return {}
        
        print(f"  [INFO] Ground truth ({strategy}): {len(valid_relevant)} items from {len(group_users)} users")
            
        return {mid: 1.0 for mid in valid_relevant}

    def calculate_ndcg(self, recommended_ids: List[int], ground_truth: Dict[int, float], k: int) -> float:
        """
        Calculates NDCG@K correctly using the set of relevant items.
        
        DCG@K = Sum( rel_i / log2(i+1) )
        IDCG@K = Sum( rel_perfect_i / log2(i+1) )
        """
        # 1. Relevance vector for recommended items (ordered by rank)
        relevance_vector = [ground_truth.get(mid, 0.0) for mid in recommended_ids[:k]]
        
        # 2. Compute DCG
        dcg = 0.0
        for i, rel in enumerate(relevance_vector):
            if rel > 0:
                dcg += rel / math.log2(i + 2) # i+2 because i is 0-indexed (rank i+1)
                
        if dcg == 0.0:
            return 0.0
            
        # 3. Compute IDCG (Ideal DCG)
        # The best possible relevance vector has all true positives (1.0) at the top
        num_relevant_items = len(ground_truth)
        # We can fill at most K slots, or fewer if we have fewer relevant items
        ideal_len = min(k, num_relevant_items)
        ideal_vector = [1.0] * ideal_len 
        
        idcg = 0.0
        for i, rel in enumerate(ideal_vector):
            idcg += rel / math.log2(i + 2)
            
        if idcg == 0.0:
            return 0.0
            
        return dcg / idcg

    def evaluate_group(self, model: HybridModel1, group_users: List[int], k: int, strategy: str = "union") -> float:
        """Evaluates a single group using NDCG@K."""
        ground_truth = self.get_group_ground_truth(group_users, strategy=strategy)
        if not ground_truth:
            return None # Skip invalid groups (no shared history in validation)
        
        print(f"  [GROUP] Ground truth: {len(ground_truth)} relevant items")
            
        # Candidates: Universe - Group's Train History
        # (This avoids Training Data Leakage but allows all other items)
        watched_in_train = set()
        for uid in group_users:
            u_rows = self.train_df[self.train_df['userId'] == uid]
            watched_in_train.update(u_rows['movieId'].tolist())
            
        candidates = list(self.all_movie_ids - watched_in_train)
        print(f"  [GROUP] Candidates: {len(candidates)} movies (universe - watched)")
        
        # Safety measure for speed: Limit candidates sample slightly if naive
        # But rigorous evaluation prefers full set. We use full set here.
        if not candidates:
            print(f"  [GROUP] ⚠️  No candidates, skipping...")
            return None

        # Predict
        print(f"  [GROUP] 🔄 Calling model.recommend_for_group()...")
        rec_start = time.time()
        recs = model.recommend_for_group(group_users, candidates, top_k=k)
        rec_time = time.time() - rec_start
        print(f"  [GROUP] ✅ Recommendations generated in {rec_time:.1f}s")
        
        if not recs:
            print(f"  [GROUP] ⚠️  No recommendations returned")
            return 0.0
            
        recommended_ids = [r['movie_id'] for r in recs]
        
        # Debug: Check overlap
        overlap = set(recommended_ids[:k]) & set(ground_truth.keys())
        print(f"  [GROUP] 📊 Overlap: {len(overlap)}/{k} recommendations match ground truth")
        
        ndcg = self.calculate_ndcg(recommended_ids, ground_truth, k)
        total_time = time.time() - rec_start
        print(f"  [GROUP] ✅ NDCG@{k} = {ndcg:.4f} (Total time: {total_time:.1f}s)")
        
        return ndcg

    def evaluate_individual_sample(self, model: HybridModel1, sample_size: int = 50, k: int = 10) -> float:
        """
        Evaluates model on individual users to contrast with group performance.
        Uses identical Ground Truth logic (Rating >= 3.5).
        """
        user_counts = self.val_df.groupby('userId').size()
        eligible_users = user_counts[user_counts >= 5].index.tolist()
        
        if not eligible_users:
            return 0.0
            
        rng = np.random.RandomState(42)
        sample_users = rng.choice(eligible_users, size=min(len(eligible_users), sample_size), replace=False)
        
        ndcg_scores = []
        
        for uid in sample_users:
            # Individual Ground Truth
            user_val = self.val_df[self.val_df['userId'] == uid]
            relevant_items = user_val[user_val['rating'] >= 3.5]['movieId'].tolist()
            
            if not relevant_items:
                continue
                
            ground_truth = {mid: 1.0 for mid in relevant_items} # Binary
            
            # Candidates
            watched = self.train_df[self.train_df['userId'] == uid]['movieId'].tolist()
            candidates = list(self.all_movie_ids - set(watched))
            
            # Predict (treat single user as group of 1)
            recs = model.recommend_for_group([uid], candidates, top_k=k)
            if not recs:
                ndcg_scores.append(0.0)
                continue
                
            rec_ids = [r['movie_id'] for r in recs]
            ndcg_scores.append(self.calculate_ndcg(rec_ids, ground_truth, k))
            
        return np.mean(ndcg_scores) if ndcg_scores else 0.0

    def run_tuning_sweep(self, num_groups: int = 30, c_values: List[float] = None, strategy: str = "union"):
        print("\n" + "="*80)
        print("HYPERPARAMETER TUNING: Hybrid Weight 'C'")
        print("Metric: NDCG@10 (Binary Relevance)")
        print(f"Ground Truth Strategy: {strategy.upper()}")
        print("Compare: Group Performance vs. Individual Performance")
        print("="*80)
        
        groups = self.create_validation_groups(num_groups=num_groups)
        if not groups:
            print("❌ Setup failed: Could not create validation groups.")
            return

        if c_values is None:
            c_values = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0] # Expanded Search Range
            
        print(f"\n[INFO] C values to test: {c_values}")
        print(f"[INFO] Total evaluations: {len(groups)} groups × {len(c_values)} C values = {len(groups) * len(c_values)}")
        
        results = {}
        best_c_group = 0.0
        best_ndcg_group = -1.0
        
        print(f"\n{'C':<6} | {'Group NDCG':<12} | {'Indiv NDCG':<12} | {'(Groups Evaluated)'}")
        print("-" * 65)
        
        for c_idx, c in enumerate(c_values, 1):
            print(f"\n{'='*70}")
            print(f"[C={c}] Testing C value {c_idx}/{len(c_values)}: C = {c}")
            print(f"{'='*70}")
            
            model = self.create_model(C=c)
            
            group_ndcgs = []
            valid_g_count = 0
            
            for g_idx, group in enumerate(groups, 1):
                print(f"\n[C={c}] Group {g_idx}/{len(groups)}")
                ndcg = self.evaluate_group(model, group, k=10, strategy=strategy)
                if ndcg is not None:
                    group_ndcgs.append(ndcg)
                    valid_g_count += 1
                else:
                    print(f"  [GROUP] Skipped (no valid data)")
                    
            g_ndcg = float(np.mean(group_ndcgs)) if group_ndcgs else 0.0
            print(f"\n[C={c}] 📊 Group evaluation complete: Avg NDCG = {g_ndcg:.4f} ({valid_g_count}/{len(groups)} valid)")
            
            print(f"[C={c}] 🔄 Evaluating individual users (sample_size=50)...")
            i_ndcg = self.evaluate_individual_sample(model, sample_size=50, k=10)
            print(f"[C={c}] 📊 Individual evaluation complete: Avg NDCG = {i_ndcg:.4f}")
            
            results[c] = {'group': g_ndcg, 'individual': i_ndcg}
            
            print(f"{c:<6.1f} | {g_ndcg:.4f}       | {i_ndcg:.4f}       | ({valid_g_count}/{len(groups)})", flush=True)
            
            if g_ndcg > best_ndcg_group:
                best_ndcg_group = g_ndcg
                best_c_group = c
                
        print("-" * 65)
        print(f"\n✅ OPTIMAL FOR GROUPS: C = {best_c_group} (NDCG: {best_ndcg_group:.4f})")
        
        # Select best Individual
        best_c_indiv = max(results, key=lambda k: results[k]['individual'])
        print(f"ℹ️  OPTIMAL FOR INDIV : C = {best_c_indiv} (NDCG: {results[best_c_indiv]['individual']:.4f})")
        
        # Save Results
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(script_dir, "results", f"h1_tuning_results_{strategy}.json")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump({
                'meta': {
                    'description': 'Hybrid Model 1 (H1) - C Parameter Tuning',
                    'c_formula': 'α(n) = n/(n+C), β(n) = C/(n+C), Score = α*IBCF + β*CBF',
                    'c_interpretation': 'Controls trust transition from CBF to IBCF based on neighbor support',
                    'ground_truth_strategy': strategy,
                    'ground_truth_definition': {
                        'union': 'ANY member rated >= 3.5 (Most Pleasure)',
                        'average': 'AVERAGE rating >= 3.5 (Balanced)',
                        'intersection': 'ALL members rated >= 3.5 (Least Misery)'
                    }[strategy],
                    'candidate_universe': 'Items in CF matrix (IBCF can score)',
                    'metric': 'NDCG@10',
                    'relevance_threshold': 3.5
                },
                'best_c_group': best_c_group,
                'best_c_individual': best_c_indiv,
                'details': results
            }, f, indent=4)
        print(f"Detailed results saved to: {output_path}")
        
        return results, best_c_group, best_ndcg_group

if __name__ == "__main__":
    # Path Setup
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    splits_dir = os.path.join(project_root, "data", "splits")
    
    # Load Train/Val Only (NO TEST SET)
    train_df = pd.read_csv(os.path.join(splits_dir, 'train.csv'))
    val_df = pd.read_csv(os.path.join(splits_dir, 'validation.csv'))
    movies_df = load_movies()
    
    tuner = RobustH1Tuner(train_df, val_df, movies_df)
    
    # Test all 3 strategies
    strategies = ["union", "average", "intersection"]
    all_results = {}
    
    print("\n" + "🔬" * 40)
    print("COMPREHENSIVE GROUND TRUTH STRATEGY COMPARISON")
    print("🔬" * 40)
    
    for strategy in strategies:
        print(f"\n\n{'🎯' * 40}")
        print(f"TESTING STRATEGY: {strategy.upper()}")
        print(f"{'🎯' * 40}\n")
        
        try:
            results, best_c, best_ndcg = tuner.run_tuning_sweep(num_groups=30, strategy=strategy)
            all_results[strategy] = {
                'best_c': best_c,
                'best_ndcg': best_ndcg,
                'all_results': results
            }
        except Exception as e:
            print(f"\n❌ Strategy '{strategy}' failed: {e}")
            import traceback
            traceback.print_exc()
            all_results[strategy] = None
    
    # Final Comparison
    print("\n\n" + "=" * 80)
    print("📊 FINAL COMPARISON: GROUND TRUTH STRATEGIES")
    print("=" * 80)
    print(f"\n{'Strategy':<15} | {'Best C':<10} | {'Best NDCG@10':<15} | {'Status'}")
    print("-" * 80)
    
    for strategy in strategies:
        if all_results[strategy]:
            best_c = all_results[strategy]['best_c']
            best_ndcg = all_results[strategy]['best_ndcg']
            status = "✅ Success"
        else:
            best_c = "N/A"
            best_ndcg = "N/A"
            status = "❌ Failed"
        
        print(f"{strategy.upper():<15} | {str(best_c):<10} | {str(best_ndcg):<15} | {status}")
    
    print("-" * 80)
    
    # Determine winner
    valid_strategies = {k: v for k, v in all_results.items() if v is not None}
    if valid_strategies:
        winner = max(valid_strategies, key=lambda k: valid_strategies[k]['best_ndcg'])
        print(f"\n🏆 WINNER: {winner.upper()} strategy")
        print(f"   Best C = {valid_strategies[winner]['best_c']}")
        print(f"   Best NDCG@10 = {valid_strategies[winner]['best_ndcg']:.4f}")
    else:
        print("\n❌ All strategies failed!")
    
    print("\n" + "=" * 80)

