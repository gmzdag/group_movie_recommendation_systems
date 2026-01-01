"""
HYBRID MODEL 2 - GROUP-LEVEL OPTIMIZATION
=========================================

RESEARCH QUESTION:
Are individual-level optimal parameters (w=0.05/0.95, AVERAGE aggregation) 
also optimal for GROUP recommendations?

METHODOLOGY: GRID SEARCH
- Test ALL combinations of weights × aggregation strategies
- Weights: w_ubcf ∈ {0.0, 0.05, 0.10, 0.15, 0.20, 0.25}
- Aggregation: {AVERAGE, LEAST_MISERY, HARMONIC_MEAN}
- Total: 6 × 3 = 18 configurations

METRIC: Group NDCG@10 (on validation groups)

HYPOTHESIS:
1. Individual-optimal weights may differ from group-optimal weights
2. Different aggregation strategies may prefer different optimal weights
   → Grid search is necessary to find truly optimal configuration

SCIENTIFIC VALIDITY:
- Minimum 30 groups required for statistical power (Amer-Yahia et al., 2009)
- Negative sampling (100 negatives per group) for robust NDCG calculation
"""

import os
import sys
import numpy as np
import pandas as pd
from functools import partial
from typing import List, Dict
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from recommender.data_loader import load_movies, load_train_valid_test_splits, build_cf_matrix
from recommender.UBCF.user_based_cf import UserBasedCF
from recommender.UBCF.neighbors_user import load_or_compute_neighbors
from recommender.UBCF.similarity_user import cosine_sim
from recommender.CB.content_based import ContentBasedModel
from sklearn.metrics import ndcg_score

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


class GroupHybrid2:
    """Configurable Hybrid 2 for group testing."""
    
    def __init__(self, ubcf, cbf, w_ubcf=0.05, w_cbf=0.95, aggregation='average'):
        self.ubcf = ubcf
        self.cbf = cbf
        self.w_ubcf = w_ubcf
        self.w_cbf = w_cbf
        self.aggregation = aggregation
    
    def predict(self, uid, mid):
        """Performance-weighted prediction."""
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
    
    def recommend_for_group(self, group_users, candidates, top_k=10):
        """Group recommendation with configurable aggregation."""
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
            
            if not member_scores:
                continue
            
            # Apply aggregation strategy
            if self.aggregation == 'average':
                final_score = np.mean(member_scores)
            elif self.aggregation == 'least_misery':
                final_score = np.min(member_scores)  # MIN
            elif self.aggregation == 'harmonic_mean':
                final_score = len(member_scores) / sum(1.0/s for s in member_scores if s > 0)
            else:
                final_score = np.mean(member_scores)  # fallback
            
            group_scores.append((mid, final_score, member_scores))
        
        group_scores.sort(key=lambda x: x[1], reverse=True)
        return group_scores[:top_k]


def create_validation_groups(valid_df, min_group_size=2, max_group_size=4, num_groups=20):
    """Create validation groups with sufficient ratings."""
    # Get users with validation ratings
    user_ratings = valid_df.groupby('userId').size()
    eligible_users = user_ratings[user_ratings >= 5].index.tolist()
    
    if len(eligible_users) < min_group_size * num_groups:
        print(f"[WARNING] Not enough users. Creating {len(eligible_users)//min_group_size} groups")
        num_groups = len(eligible_users) // min_group_size
    
    # Random group formation
    np.random.seed(42)
    np.random.shuffle(eligible_users)
    
    groups = []
    for i in range(num_groups):
        group_size = np.random.randint(min_group_size, max_group_size + 1)
        start_idx = i * min_group_size
        end_idx = start_idx + group_size
        
        if end_idx > len(eligible_users):
            break
        
        group = eligible_users[start_idx:end_idx]
        groups.append(group)
    
    return groups


def evaluate_group_ndcg(model, groups, valid_df, R_train, k=10):
    """Calculate Group NDCG@K using Negative Sampling (100 negatives)."""
    group_ndcgs = []
    group_fairness = []
    
    all_items = R_train.columns.tolist()
    all_items_set = set(all_items)
    
    rng = np.random.RandomState(42)  # Consistent sampling

    for group_users in groups:
        # Get group's validation ratings
        group_valid = valid_df[valid_df['userId'].isin(group_users)]
        
        # Ground Truth: Items rated in validation by ANY member
        relevant_items = group_valid['movieId'].unique().tolist()
        
        if not relevant_items:
             continue
             
        # Get watched in TRAIN (to exclude from negatives)
        watched_train = set()
        for uid in group_users:
            if uid in R_train.index:
                watched_train.update(R_train.loc[uid].dropna().index)
        
        # Negatives Universe: All items - Watched_Train - Relevant_Validation
        negatives_pool = list(all_items_set - watched_train - set(relevant_items))
        
        # Sample 100 Negatives
        if len(negatives_pool) > 100:
            negatives = list(rng.choice(negatives_pool, size=100, replace=False))
        else:
            negatives = negatives_pool
            
        # Candidates = Relevant + Negatives
        candidates = list(set(relevant_items + negatives))
        
        # Get recommendations for ALL candidates (sorted by score)
        try:
            recs = model.recommend_for_group(group_users, candidates, top_k=len(candidates))
        except:
            continue
        
        if not recs:
            continue
        
        # Prepare arrays for sklearn ndcg_score
        y_true = []
        y_score = []
        
        all_member_scores_top_k = [] # For fairness (on top k)
        
        for i, (mid, score, member_scores) in enumerate(recs):
            # Ground truth: average of members' validation ratings for this item
            item_ratings = group_valid[group_valid['movieId'] == mid]['rating'].tolist()
            if item_ratings:
                # Use binary relevance or actual rating? Hybrid 1 used binary 1.0
                # But Hybrid 2 typically uses ratings. Let's stick to ratings for precision.
                # If rating >= 3.5 it is relevant? 
                # Sklearn handles graded relevance.
                avg_rating = np.mean(item_ratings)
            else:
                avg_rating = 0.0
            
            y_true.append(avg_rating)
            y_score.append(score)
            
            if i < k:
                all_member_scores_top_k.extend(member_scores)

        # Calculate NDCG@k
        # We need at least one positive relevant item to be meaningful
        if sum(y_true) > 0:
            try:
                ndcg = ndcg_score([y_true], [y_score], k=k)
                group_ndcgs.append(ndcg)
                
                # Fairness on Top K items only
                if all_member_scores_top_k:
                    # Avoid division by zero
                    mean_val = np.mean(all_member_scores_top_k)
                    if mean_val > 0:
                        fairness = np.min(all_member_scores_top_k) / mean_val
                        group_fairness.append(fairness)
            except:
                pass
    
    avg_ndcg = np.mean(group_ndcgs) if group_ndcgs else 0.0
    avg_fairness = np.mean(group_fairness) if group_fairness else 0.0
    num_groups_evaluated = len(group_ndcgs)
    
    return avg_ndcg, avg_fairness, num_groups_evaluated


def main():
    print("="*70)
    print(" HYBRID MODEL 2 - GROUP-LEVEL OPTIMIZATION")
    print(" Testing Weights and Aggregation Strategies")
    print("="*70)
    
    # Load data
    print("\n[1] Loading Data...")
    movies = load_movies()
    train_df, valid_df, test_df = load_train_valid_test_splits()
    
    print(f"  Train: {len(train_df):,} ratings")
    print(f"  Valid: {len(valid_df):,} ratings")
    
    # Build models
    print("\n[2] Building Base Models...")
    R_train = build_cf_matrix(train_df)
    user_means = R_train.mean(axis=1)
    item_means = R_train.mean(axis=0)
    global_mean = train_df["rating"].mean()
    
    print("  UBCF (Cosine, K=50, Overlap=5)...")
    # Reuse existing cache from hybrid2_threshold_optimization
    neighbors = load_or_compute_neighbors(
        R_train,
        partial(cosine_sim, MIN_OVERLAP=5),
        K=50,
        metric="cosine_overlap5_hybrid2_opt"  # ✅ REUSE EXISTING CACHE
    )
    ubcf = UserBasedCF(R_train, neighbors, user_means, item_means, global_mean, movies=movies)
    
    print("  CBF (Optimized Weights)...")
    cbf = ContentBasedModel(movies, train_df)
    
    # Create validation groups
    print("\n[3] Creating Validation Groups...")
    # Literature standard: min 30 groups for statistical validity (Amer-Yahia et al., 2009)
    groups = create_validation_groups(valid_df, min_group_size=2, max_group_size=4, num_groups=30)
    print(f"  Created {len(groups)} groups (sizes: {[len(g) for g in groups]})")
    
    # Scientific validity check
    if len(groups) < 30:
        print(f"  ⚠️  WARNING: Only {len(groups)} groups created (literature standard: ≥30)")
        print(f"  Results may lack statistical power (Amer-Yahia et al., 2009)")
    else:
        print(f"  ✅ Sufficient groups for statistical validity (≥30)")
    
    # GRID SEARCH: Test ALL combinations of weights × aggregation strategies
    print("\n[4] GRID SEARCH: Weights × Aggregation Strategies")
    print("="*70)
    print("\n  Testing all combinations to find truly optimal configuration...")
    print("  (Each aggregation may have different optimal weights)")
    
    all_results = []
    
    weight_values = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
    aggregation_strategies = ['average', 'least_misery', 'harmonic_mean']
    
    total_tests = len(weight_values) * len(aggregation_strategies)
    test_count = 0
    
    for agg_strategy in aggregation_strategies:
        print(f"\n  {'='*66}")
        print(f"  AGGREGATION: {agg_strategy.upper()}")
        print(f"  {'='*66}")
        
        for w_ubcf in weight_values:
            w_cbf = 1.0 - w_ubcf
            test_count += 1
            
            print(f"\n  [{test_count}/{total_tests}] w_ubcf={w_ubcf:.2f}, w_cbf={w_cbf:.2f}, agg={agg_strategy}")
            
            model = GroupHybrid2(ubcf, cbf, w_ubcf=w_ubcf, w_cbf=w_cbf, aggregation=agg_strategy)
            ndcg, fairness, num_groups = evaluate_group_ndcg(model, groups, valid_df, R_train, k=10)
            
            print(f"      → Group NDCG@10: {ndcg:.4f}, Fairness: {fairness:.4f}, Groups: {num_groups}")
            
            all_results.append({
                'w_ubcf': w_ubcf,
                'w_cbf': w_cbf,
                'aggregation': agg_strategy,
                'group_ndcg': ndcg,
                'fairness': fairness,
                'num_groups': num_groups
            })
    
    # Results Analysis
    print("\n" + "="*70)
    print(" GRID SEARCH RESULTS")
    print("="*70)
    
    # Create DataFrame for analysis
    df_results = pd.DataFrame(all_results)
    
    # Show results grouped by aggregation
    print("\n1. RESULTS BY AGGREGATION STRATEGY:")
    print("-" * 70)
    
    for agg in aggregation_strategies:
        agg_data = df_results[df_results['aggregation'] == agg].sort_values('group_ndcg', ascending=False)
        best_for_agg = agg_data.iloc[0]
        
        print(f"\n  {agg.upper()}:")
        print(f"    Best weights: w_ubcf={best_for_agg['w_ubcf']:.2f}, w_cbf={best_for_agg['w_cbf']:.2f}")
        print(f"    Group NDCG@10: {best_for_agg['group_ndcg']:.4f}")
        print(f"    Fairness: {best_for_agg['fairness']:.4f}")
        print(f"\n    All results for {agg}:")
        print(agg_data[['w_ubcf', 'w_cbf', 'group_ndcg', 'fairness']].to_string(index=False))
    
    # Find overall best configuration
    best_overall = df_results.loc[df_results['group_ndcg'].idxmax()]
    
    print("\n" + "="*70)
    print(" OVERALL BEST CONFIGURATION")
    print("="*70)
    print(f"\n  Aggregation: {best_overall['aggregation'].upper()}")
    print(f"  w_ubcf: {best_overall['w_ubcf']:.2f}")
    print(f"  w_cbf: {best_overall['w_cbf']:.2f}")
    print(f"  Group NDCG@10: {best_overall['group_ndcg']:.4f}")
    print(f"  Fairness: {best_overall['fairness']:.4f}")
    print(f"  Groups Evaluated: {int(best_overall['num_groups'])}")
    
    # Comparison with individual-optimal (w=0.05/0.95, AVERAGE)
    print("\n" + "="*70)
    print(" COMPARISON: Individual-Optimal vs Group-Optimal")
    print("="*70)
    
    individual_optimal = df_results[
        (df_results['w_ubcf'] == 0.05) & 
        (df_results['aggregation'] == 'average')
    ]
    
    if not individual_optimal.empty:
        ind_opt = individual_optimal.iloc[0]
        print(f"\nIndividual-Optimal (from literature, w=0.05/0.95, AVERAGE):")
        print(f"  Group NDCG@10: {ind_opt['group_ndcg']:.4f}")
        print(f"  Fairness: {ind_opt['fairness']:.4f}")
        
        print(f"\nGroup-Optimal (from grid search):")
        print(f"  Aggregation: {best_overall['aggregation'].upper()}")
        print(f"  Weights: w_ubcf={best_overall['w_ubcf']:.2f}, w_cbf={best_overall['w_cbf']:.2f}")
        print(f"  Group NDCG@10: {best_overall['group_ndcg']:.4f}")
        print(f"  Fairness: {best_overall['fairness']:.4f}")
        
        improvement = ((best_overall['group_ndcg'] / ind_opt['group_ndcg']) - 1) * 100
        print(f"\nImprovement: {improvement:+.2f}%")
        
        if abs(improvement) < 1.0:
            print("\n✅ Individual-optimal is also group-optimal (no significant change)")
        else:
            print(f"\n⚠️  Group-specific optimization provides {abs(improvement):.2f}% improvement")
            print("   📌 RECOMMENDATION: Use group-optimal configuration for production")
    
    # Key Insights
    print("\n" + "="*70)
    print(" KEY INSIGHTS")
    print("="*70)
    
    # Check if different aggregations prefer different weights
    best_per_agg = df_results.groupby('aggregation').apply(
        lambda x: x.loc[x['group_ndcg'].idxmax()]
    )
    
    unique_weights = best_per_agg[['w_ubcf', 'w_cbf']].drop_duplicates()
    
    if len(unique_weights) > 1:
        print("\n  🔍 Different aggregation strategies prefer DIFFERENT optimal weights:")
        for agg in aggregation_strategies:
            best = best_per_agg.loc[agg]
            print(f"     • {agg.upper()}: w_ubcf={best['w_ubcf']:.2f}")
        print("\n  ✅ Grid search approach was NECESSARY!")
    else:
        print("\n  ℹ️  All aggregation strategies prefer the SAME optimal weights")
        print(f"     w_ubcf={best_per_agg.iloc[0]['w_ubcf']:.2f}")
        print("\n  → Sequential optimization would have worked in this case")
    
    # Save results
    csv_path = os.path.join(RESULTS_DIR, "hybrid2_group_optimization.csv")
    pd.DataFrame(all_results).to_csv(csv_path, index=False)
    print(f"\n💾 Saved: {csv_path}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
