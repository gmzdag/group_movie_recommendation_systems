"""
HYBRID MODEL 2 - GROUP-LEVEL OPTIMIZATION
=========================================

RESEARCH QUESTION:
Are individual-level optimal parameters (w=0.05/0.95, AVERAGE aggregation) 
also optimal for GROUP recommendations?

TESTS:
1. Weight Optimization: w_ubcf ∈ {0.0, 0.05, 0.10, 0.15, 0.20, 0.25}
2. Aggregation Strategy: AVERAGE vs LEAST_MISERY vs HARMONIC_MEAN

METRIC: Group NDCG@10 (on validation groups)

HYPOTHESIS:
Individual-optimal weights may differ from group-optimal weights due to:
- Group dynamics
- Consensus effects
- Different optimization metric (Group NDCG vs Individual NDCG)

SCIENTIFIC VALIDITY:
- Minimum 30 groups required for statistical power (Amer-Yahia et al., 2009)
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

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
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
    
    # TEST 1: Weight Optimization
    print("\n[4] TEST 1: Weight Optimization (Group NDCG)")
    print("="*70)
    
    weight_results = []
    aggregation = 'average'  # Use AVERAGE for weight test
    
    for w_ubcf in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]:
        w_cbf = 1.0 - w_ubcf
        
        print(f"\n  Testing w_ubcf={w_ubcf:.2f}, w_cbf={w_cbf:.2f}...")
        
        model = GroupHybrid2(ubcf, cbf, w_ubcf=w_ubcf, w_cbf=w_cbf, aggregation=aggregation)
        ndcg, fairness, num_groups = evaluate_group_ndcg(model, groups, valid_df, R_train, k=10)
        
        print(f"    Group NDCG@10: {ndcg:.4f}")
        print(f"    Fairness: {fairness:.4f}")
        print(f"    Groups Evaluated: {num_groups}")
        
        weight_results.append({
            'w_ubcf': w_ubcf,
            'w_cbf': w_cbf,
            'aggregation': aggregation,
            'group_ndcg': ndcg,
            'fairness': fairness,
            'num_groups': num_groups
        })
    
    # TEST 2: Aggregation Strategy (with optimal weight from TEST 1)
    print("\n[5] TEST 2: Aggregation Strategy Comparison")
    print("="*70)
    
    # Find best weight from TEST 1
    best_weight_config = max(weight_results, key=lambda x: x['group_ndcg'])
    optimal_w_ubcf = best_weight_config['w_ubcf']
    optimal_w_cbf = best_weight_config['w_cbf']
    
    print(f"\n  Using optimal weight: w_ubcf={optimal_w_ubcf:.2f}, w_cbf={optimal_w_cbf:.2f}")
    
    aggregation_results = []
    
    for agg_strategy in ['average', 'least_misery', 'harmonic_mean']:
        print(f"\n  Testing {agg_strategy.upper()}...")
        
        model = GroupHybrid2(ubcf, cbf, w_ubcf=optimal_w_ubcf, w_cbf=optimal_w_cbf, 
                           aggregation=agg_strategy)
        ndcg, fairness, num_groups = evaluate_group_ndcg(model, groups, valid_df, R_train, k=10)
        
        print(f"    Group NDCG@10: {ndcg:.4f}")
        print(f"    Fairness: {fairness:.4f}")
        print(f"    Groups Evaluated: {num_groups}")
        
        aggregation_results.append({
            'w_ubcf': optimal_w_ubcf,
            'w_cbf': optimal_w_cbf,
            'aggregation': agg_strategy,
            'group_ndcg': ndcg,
            'fairness': fairness,
            'num_groups': num_groups
        })
    
    # Results Analysis
    print("\n" + "="*70)
    print(" OPTIMIZATION RESULTS")
    print("="*70)
    
    print("\n1. WEIGHT OPTIMIZATION:")
    df_weights = pd.DataFrame(weight_results)
    df_weights_sorted = df_weights.sort_values('group_ndcg', ascending=False)
    print(df_weights_sorted.to_string(index=False))
    
    print("\n2. AGGREGATION STRATEGY:")
    df_agg = pd.DataFrame(aggregation_results)
    df_agg_sorted = df_agg.sort_values('group_ndcg', ascending=False)
    print(df_agg_sorted.to_string(index=False))
    
    # Find overall best
    all_results = weight_results + aggregation_results
    best_overall = max(all_results, key=lambda x: x['group_ndcg'])
    
    print("\n" + "="*70)
    print(" BEST CONFIGURATION FOR GROUPS")
    print("="*70)
    print(f"\n  w_ubcf: {best_overall['w_ubcf']:.2f}")
    print(f"  w_cbf: {best_overall['w_cbf']:.2f}")
    print(f"  Aggregation: {best_overall['aggregation'].upper()}")
    print(f"  Group NDCG@10: {best_overall['group_ndcg']:.4f}")
    print(f"  Fairness: {best_overall['fairness']:.4f}")
    
    # Comparison with individual-optimal
    print("\n" + "="*70)
    print(" COMPARISON: Individual vs Group Optimization")
    print("="*70)
    
    individual_optimal = next((r for r in all_results if r['w_ubcf'] == 0.05 and r['aggregation'] == 'average'), None)
    
    if individual_optimal:
        print(f"\nIndividual-Optimal (w=0.05/0.95, AVERAGE):")
        print(f"  Group NDCG@10: {individual_optimal['group_ndcg']:.4f}")
        
        print(f"\nGroup-Optimal:")
        print(f"  Group NDCG@10: {best_overall['group_ndcg']:.4f}")
        
        improvement = ((best_overall['group_ndcg'] / individual_optimal['group_ndcg']) - 1) * 100
        print(f"\nImprovement: {improvement:+.1f}%")
        
        if abs(improvement) < 2.0:
            print("\n✅ Individual-optimal is also group-optimal (no change needed)")
        else:
            print(f"\n⚠️ Group-specific optimization provides {abs(improvement):.1f}% improvement")
            print("   Recommendation: Use group-optimal configuration")
    
    # Save results
    csv_path = os.path.join(RESULTS_DIR, "hybrid2_group_optimization.csv")
    pd.DataFrame(all_results).to_csv(csv_path, index=False)
    print(f"\n💾 Saved: {csv_path}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
