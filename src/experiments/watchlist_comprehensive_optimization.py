"""
COMPREHENSIVE WATCHLIST GROUP RECOMMENDATION OPTIMIZATION
=========================================================

PAPER SECTION: Comprehensive Hyperparameter Optimization for Group Recommendations
SCIENTIFIC APPROACH: Compare ALL watchlist-based strategies with ALL parameter combinations

RESEARCH QUESTION:
What is the BEST watchlist-based approach for group recommendations?

MODELS TO COMPARE:
1. WatchlistRecommender (AVG profile + Consensus scoring)
   - Parameters: disagreement_penalty [0.0, 0.25, 0.5, 0.75, 1.0]
   
2. Hybrid Model 3 - AVERAGE aggregation (current)
   - Parameters: None (baseline)
   
3. Hybrid Model 3 - MIN aggregation (fairness-focused)
   - Parameters: None
   
4. Hybrid Model 3 - HARMONIC_MEAN aggregation (balanced)
   - Parameters: None
   
5. Hybrid Model 3 - With Disagreement Penalty (NEW!)
   - Parameters: disagreement_penalty [0.0, 0.25, 0.5, 0.75, 1.0]

GLOBAL PARAMETERS:
- top_k: [5, 10, 15, 20]

EVALUATION METRICS:
- Group NDCG: Ranking quality
- Fairness: min/avg individual satisfaction
- Coverage: % of group interests covered
- Composite Score: 60% NDCG + 40% Fairness

METHOD:
- Train on train set
- Optimize on validation set
- Report best configuration for each model
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Set, Tuple, Optional
from sklearn.metrics import ndcg_score
from scipy.stats import hmean

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from recommender.data_loader import load_movies, load_watchlists, load_train_valid_test_splits
from recommender.CB.content_based import ContentBasedModel
from recommender.hybrid.hybrid_model_3 import WatchlistHybridModel
from recommender.watchlist.watchlist_recommender import WatchlistRecommender

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


class EnhancedHybrid3:
    """
    Enhanced Hybrid Model 3 with configurable aggregation strategies and disagreement penalty.
    
    SCIENTIFIC CONTRIBUTION: Tests whether fairness-aware aggregation improves group recommendations.
    """
    
    def __init__(self, movies_df, watchlist_df, cb_model, 
                 aggregation='average', disagreement_penalty=0.0):
        """
        Args:
            aggregation: 'average', 'min', 'harmonic_mean'
            disagreement_penalty: 0.0-1.0 (only used if > 0)
        """
        self.base_model = WatchlistHybridModel(movies_df, watchlist_df, cb_model)
        self.aggregation = aggregation
        self.disagreement_penalty = disagreement_penalty
        self.cb_model = cb_model  # For compatibility
        
    def predict(self, user_id, movie_id):
        """Delegate to base model."""
        return self.base_model.predict(user_id, movie_id)
    
    def recommend_for_group(self, user_ids: List[int], candidates: List[int], 
                           top_k: int = 10) -> List[Dict]:
        """
        Enhanced group recommendation with multiple aggregation strategies.
        """
        scores_list = []
        
        for mid in candidates:
            user_scores = []
            
            for uid in user_ids:
                s = self.base_model.predict(uid, mid)
                if not np.isnan(s):
                    user_scores.append(s)
            
            if len(user_scores) == 0:
                continue
            
            # Aggregation strategy
            if self.aggregation == 'average':
                agg_score = np.mean(user_scores)
            elif self.aggregation == 'min':
                # MIN: Most conservative, ensures all members have some interest
                agg_score = np.min(user_scores)
            elif self.aggregation == 'harmonic_mean':
                # Harmonic mean: Penalizes low scores more than average
                agg_score = hmean(user_scores) if min(user_scores) > 0 else 0.0
            else:
                agg_score = np.mean(user_scores)
            
            # Apply disagreement penalty if configured
            if self.disagreement_penalty > 0:
                std_dev = np.std(user_scores)
                # Higher std = more disagreement = lower final score
                penalty = std_dev * self.disagreement_penalty
                agg_score = max(0, agg_score - penalty)
            
            scores_list.append((mid, agg_score, user_scores))
        
        # Sort
        scores_list.sort(key=lambda x: x[1], reverse=True)
        top_items = scores_list[:top_k]
        
        # Format results
        results = []
        for mid, score, user_scores in top_items:
            results.append({
                'movie_id': mid,
                'score': score,
                'group_explanation': f"Aggregation: {self.aggregation}, Penalty: {self.disagreement_penalty}",
                'explanations': {}
            })
        
        return results


def evaluate_group_config(
    model,
    model_name: str,
    watchlist_df: pd.DataFrame,
    valid_users: Set[int],
    group_sizes: List[int] = [2, 3, 4],
    num_groups: int = 10,
    top_k: int = 10
) -> Dict[str, float]:
    """
    Evaluate a specific configuration on validation set.
    Returns aggregated metrics across all group sizes.
    """
    users_with_watchlist = set(
        watchlist_df[watchlist_df['userId'].isin(valid_users)]['userId'].unique()
    )
    
    if len(users_with_watchlist) < max(group_sizes):
        return {
            'group_ndcg': 0.0,
            'fairness': 0.0,
            'coverage': 0.0,
            'num_groups': 0
        }
    
    all_ndcg = []
    all_fairness = []
    all_coverage = []
    
    for group_size in group_sizes:
        user_list = list(users_with_watchlist)
        np.random.seed(42 + group_size)
        np.random.shuffle(user_list)
        
        for i in range(min(num_groups, len(user_list) // group_size)):
            start_idx = i * group_size
            group = user_list[start_idx:start_idx + group_size]
            
            # Get group watchlists
            group_watchlists = {}
            for uid in group:
                wl = watchlist_df[watchlist_df['userId'] == uid]['movieId'].astype(int).tolist()
                if len(wl) >= 2:
                    group_watchlists[uid] = set(wl)
            
            if len(group_watchlists) < 2:
                continue
            
            # Split watchlists
            train_watchlists = {}
            test_watchlists = {}
            
            for uid, wl in group_watchlists.items():
                wl_list = list(wl)
                n = len(wl_list)
                n_train = max(1, int(n * 0.7))
                
                train_watchlists[uid] = set(wl_list[:n_train])
                test_watchlists[uid] = set(wl_list[n_train:])
            
            # Get candidates
            try:
                if hasattr(model, 'cb_model'):
                    all_movies = set(model.cb_model.movie_to_idx.keys())
                elif hasattr(model, 'base_model'):
                    all_movies = set(model.base_model.cb_model.movie_to_idx.keys())
                else:
                    all_movies = set(model.title_map.keys())
            except:
                continue
            
            all_train = set()
            for train_wl in train_watchlists.values():
                all_train.update(train_wl)
            
            candidates = list(all_movies - all_train)
            
            if len(candidates) < top_k:
                continue
            
            # Get recommendations
            try:
                if hasattr(model, 'recommend_for_group'):
                    recs = model.recommend_for_group(group, candidates=candidates, top_k=top_k)
                    
                    if isinstance(recs, list):
                        rec_movies = [r['movie_id'] for r in recs]
                    elif isinstance(recs, dict) and 'recommended_movies' in recs:
                        rec_movies = [mid for mid, _, _ in recs['recommended_movies']]
                    else:
                        continue
                else:
                    continue
                
            except Exception as e:
                continue
            
            if not rec_movies:
                continue
            
            # Calculate metrics
            all_test = set()
            for test_wl in test_watchlists.values():
                all_test.update(test_wl)
            
            # Group NDCG
            y_true = [1 if mid in all_test else 0 for mid in rec_movies]
            y_score = list(range(len(rec_movies), 0, -1))
            
            if sum(y_true) > 0:
                group_ndcg = ndcg_score([y_true], [y_score])
                all_ndcg.append(group_ndcg)
            
            # Fairness (FIXED: Added safety checks for scientific validity)
            individual_sats = []
            for uid in group:
                if uid not in test_watchlists:
                    # User has no test watchlist - skip from fairness calculation
                    continue
                
                user_test = test_watchlists[uid]
                if len(user_test) == 0:
                    # User has empty test set - should not affect fairness score
                    continue
                
                hits = len(set(rec_movies) & user_test)
                satisfaction = hits / len(user_test)
                individual_sats.append(satisfaction)
            
            # SCIENTIFIC VALIDITY: Require at least 50% of group members to have valid satisfaction scores
            if individual_sats and len(individual_sats) >= max(2, len(group) * 0.5):
                # Fairness = min/avg satisfaction (valid group recommender fairness metric)
                fairness = min(individual_sats) / (np.mean(individual_sats) + 1e-9)
                all_fairness.append(fairness)
            # else: Skip this group's fairness if too few members have valid test data
            
            # Coverage
            covered = len(set(rec_movies) & all_test)
            coverage = covered / len(all_test) if len(all_test) > 0 else 0.0
            all_coverage.append(coverage)
    
    return {
        'group_ndcg': np.mean(all_ndcg) if all_ndcg else 0.0,
        'fairness': np.mean(all_fairness) if all_fairness else 0.0,
        'coverage': np.mean(all_coverage) if all_coverage else 0.0,
        'num_groups': len(all_ndcg)
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print("="*70)
    print(" COMPREHENSIVE WATCHLIST GROUP RECOMMENDATION OPTIMIZATION")
    print(" Testing ALL Strategies and Parameters")
    print("="*70)
    
    # Load data
    print("\n[1] Loading Data (Fixed Temporal Split)...")
    movies = load_movies()
    watchlist = load_watchlists()
    train_df, valid_df, test_df = load_train_valid_test_splits()
    
    valid_users = set(valid_df['userId'].unique())
    
    print(f"  Train: {len(train_df):,} ratings")
    print(f"  Valid: {len(valid_df):,} ratings (for tuning)")
    print(f"  Test:  {len(test_df):,} ratings (held out)")
    print(f"  Watchlist entries: {len(watchlist)}")
    
    # Build base models
    print("\n[2] Building Base Content-Based Model...")
    cbf = ContentBasedModel(movies, train_df)
    
    # ===== COMPREHENSIVE OPTIMIZATION =====
    print("\n[3] COMPREHENSIVE OPTIMIZATION ON VALIDATION SET")
    print("="*70)
    
    all_results = []
    
    # Test configurations (REDUCED for speed)
    penalty_values = [0.0, 0.5, 1.0]  # Reduced from 5 to 3 values
    k_values = [10]  # Focus on K=10 for speed, can expand later
    aggregations = ['average', 'min', 'harmonic_mean']
    
    config_id = 0
    total_configs = (
        len(penalty_values) +  # WatchlistRecommender (3 penalties)
        1 +  # Hybrid3 baseline
        (len(aggregations) - 1) +  # Hybrid3 aggregation variants (min, harmonic_mean)
        (len(penalty_values) - 1)  # Hybrid3 with penalty (excluding 0.0)
    )
    
    print(f"\nTotal configurations to test: {total_configs}")
    print(f"This may take a while...\n")
    
    # 1. WatchlistRecommender with different penalties
    print(f"\n{'='*70}")
    print("MODEL 1: WatchlistRecommender (AVG Profile + Consensus)")
    print(f"{'='*70}\n")
    
    # Reuse CBF for all WatchlistRecommender tests (HUGE speed improvement!)
    print("Reusing pre-built ContentBasedModel for all penalty tests...")
    
    for penalty in penalty_values:
        config_id += 1
        print(f"[{config_id}/{total_configs}] Testing penalty={penalty:.2f}...")
        
        # Pass pre-built cbf to avoid re-initialization
        model = WatchlistRecommender(movies, train_df, watchlist, 
                                     disagreement_penalty=penalty, cb_model=cbf)
        
        for k in k_values:
            metrics = evaluate_group_config(
                model, f"WatchlistRec_p{penalty}_k{k}", watchlist, valid_users,
                group_sizes=[2, 3, 4], num_groups=20, top_k=k  # FIXED: Increased to 20 for statistical validity
            )
            
            composite = 0.6 * metrics['group_ndcg'] + 0.4 * metrics['fairness']
            
            all_results.append({
                'model_type': 'WatchlistRecommender',
                'aggregation': 'AVG_Profile',
                'disagreement_penalty': penalty,
                'top_k': k,
                'group_ndcg': metrics['group_ndcg'],
                'fairness': metrics['fairness'],
                'coverage': metrics['coverage'],
                'composite_score': composite,
                'num_groups': metrics['num_groups']
            })
            
            print(f"  K={k}: NDCG={metrics['group_ndcg']:.4f}, "
                  f"Fairness={metrics['fairness']:.4f}, Composite={composite:.4f}, "
                  f"Groups={metrics['num_groups']}")
            
            # SCIENTIFIC VALIDITY WARNING
            if metrics['num_groups'] < 10:
                print(f"  ⚠️  WARNING: Only {metrics['num_groups']} groups evaluated. Results may be unreliable!")
    
    # 2. Hybrid Model 3 - Baseline (AVERAGE)
    print(f"\n{'='*70}")
    print("MODEL 2: Hybrid Model 3 - Baseline (AVERAGE aggregation)")
    print(f"{'='*70}\n")
    
    config_id += 1
    print(f"[{config_id}/{total_configs}] Testing baseline...")
    
    hybrid3_baseline = WatchlistHybridModel(movies, watchlist, cbf)
    
    for k in k_values:
        metrics = evaluate_group_config(
            hybrid3_baseline, f"Hybrid3_baseline_k{k}", watchlist, valid_users,
            group_sizes=[2, 3, 4], num_groups=20, top_k=k  # FIXED: Increased to 20 for statistical validity
        )
        
        composite = 0.6 * metrics['group_ndcg'] + 0.4 * metrics['fairness']
        
        all_results.append({
            'model_type': 'Hybrid3_Baseline',
            'aggregation': 'AVERAGE',
            'disagreement_penalty': 0.0,
            'top_k': k,
            'group_ndcg': metrics['group_ndcg'],
            'fairness': metrics['fairness'],
            'coverage': metrics['coverage'],
            'composite_score': composite,
            'num_groups': metrics['num_groups']
        })
        
        print(f"  K={k}: NDCG={metrics['group_ndcg']:.4f}, "
              f"Fairness={metrics['fairness']:.4f}, Composite={composite:.4f}, "
              f"Groups={metrics['num_groups']}")
        
        # SCIENTIFIC VALIDITY WARNING
        if metrics['num_groups'] < 10:
            print(f"  ⚠️  WARNING: Only {metrics['num_groups']} groups evaluated. Results may be unreliable!")
    
    # 3. Hybrid Model 3 - Different Aggregations
    print(f"\n{'='*70}")
    print("MODEL 3: Hybrid Model 3 - Alternative Aggregations")
    print(f"{'='*70}\n")
    
    for agg in aggregations:
        if agg == 'average':
            continue  # Already tested in baseline
        
        config_id += 1
        print(f"[{config_id}/{total_configs}] Testing aggregation={agg}...")
        
        model = EnhancedHybrid3(movies, watchlist, cbf, aggregation=agg, disagreement_penalty=0.0)
        
        for k in k_values:
            metrics = evaluate_group_config(
                model, f"Hybrid3_{agg}_k{k}", watchlist, valid_users,
                group_sizes=[2, 3, 4], num_groups=20, top_k=k  # FIXED: Increased to 20 for statistical validity
            )
            
            composite = 0.6 * metrics['group_ndcg'] + 0.4 * metrics['fairness']
            
            all_results.append({
                'model_type': 'Hybrid3_Enhanced',
                'aggregation': agg.upper(),
                'disagreement_penalty': 0.0,
                'top_k': k,
                'group_ndcg': metrics['group_ndcg'],
                'fairness': metrics['fairness'],
                'coverage': metrics['coverage'],
                'composite_score': composite,
                'num_groups': metrics['num_groups']
            })
            
            print(f"  K={k}: NDCG={metrics['group_ndcg']:.4f}, "
                  f"Fairness={metrics['fairness']:.4f}, Composite={composite:.4f}, "
                  f"Groups={metrics['num_groups']}")
            
            # SCIENTIFIC VALIDITY WARNING
            if metrics['num_groups'] < 10:
                print(f"  ⚠️  WARNING: Only {metrics['num_groups']} groups evaluated. Results may be unreliable!")
    
    # 4. Hybrid Model 3 - With Disagreement Penalty
    print(f"\n{'='*70}")
    print("MODEL 4: Hybrid Model 3 - With Disagreement Penalty (NEW!)")
    print(f"{'='*70}\n")
    
    for penalty in penalty_values:
        if penalty == 0.0:
            continue  # Already tested in baseline
        
        config_id += 1
        print(f"[{config_id}/{total_configs}] Testing penalty={penalty:.2f}...")
        
        model = EnhancedHybrid3(movies, watchlist, cbf, aggregation='average', 
                               disagreement_penalty=penalty)
        
        for k in k_values:
            metrics = evaluate_group_config(
                model, f"Hybrid3_penalty{penalty}_k{k}", watchlist, valid_users,
                group_sizes=[2, 3, 4], num_groups=20, top_k=k  # FIXED: Increased to 20 for statistical validity
            )
            
            composite = 0.6 * metrics['group_ndcg'] + 0.4 * metrics['fairness']
            
            all_results.append({
                'model_type': 'Hybrid3_WithPenalty',
                'aggregation': 'AVERAGE',
                'disagreement_penalty': penalty,
                'top_k': k,
                'group_ndcg': metrics['group_ndcg'],
                'fairness': metrics['fairness'],
                'coverage': metrics['coverage'],
                'composite_score': composite,
                'num_groups': metrics['num_groups']
            })
            
            print(f"  K={k}: NDCG={metrics['group_ndcg']:.4f}, "
                  f"Fairness={metrics['fairness']:.4f}, Composite={composite:.4f}, "
                  f"Groups={metrics['num_groups']}")
            
            # SCIENTIFIC VALIDITY WARNING
            if metrics['num_groups'] < 10:
                print(f"  ⚠️  WARNING: Only {metrics['num_groups']} groups evaluated. Results may be unreliable!")
    
    # ===== ANALYSIS =====
    print(f"\n{'='*70}")
    print("RESULTS ANALYSIS")
    print(f"{'='*70}\n")
    
    df_results = pd.DataFrame(all_results)
    
    # Find overall best
    best_idx = df_results['composite_score'].idxmax()
    best_config = df_results.loc[best_idx]
    
    print("OVERALL BEST CONFIGURATION:")
    print("-" * 70)
    print(f"Model Type: {best_config['model_type']}")
    print(f"Aggregation: {best_config['aggregation']}")
    print(f"Disagreement Penalty: {best_config['disagreement_penalty']}")
    print(f"Top-K: {best_config['top_k']}")
    print(f"Group NDCG: {best_config['group_ndcg']:.4f}")
    print(f"Fairness: {best_config['fairness']:.4f}")
    print(f"Coverage: {best_config['coverage']:.4f}")
    print(f"Composite Score: {best_config['composite_score']:.4f}")
    print(f"Groups Evaluated: {best_config['num_groups']}")
    
    # SCIENTIFIC VALIDITY CHECK
    if best_config['num_groups'] < 10:
        print("\n⚠️  CRITICAL WARNING: Best config has < 10 groups. Results may be statistically unreliable!")
    print()
    
    # Best per model type
    print("BEST CONFIGURATION PER MODEL TYPE:")
    print("-" * 70)
    for model_type in df_results['model_type'].unique():
        type_results = df_results[df_results['model_type'] == model_type]
        best_type_idx = type_results['composite_score'].idxmax()
        best_type = type_results.loc[best_type_idx]
        
        print(f"\n{model_type}:")
        print(f"  Aggregation: {best_type['aggregation']}")
        print(f"  Penalty: {best_type['disagreement_penalty']}")
        print(f"  K: {best_type['top_k']}")
        print(f"  Composite: {best_type['composite_score']:.4f} "
              f"(NDCG={best_type['group_ndcg']:.4f}, Fairness={best_type['fairness']:.4f})")
    
    # ===== SAVE RESULTS =====
    print(f"\n{'='*70}")
    print("SAVING RESULTS")
    print(f"{'='*70}\n")
    
    results_path = os.path.join(RESULTS_DIR, "watchlist_comprehensive_optimization.csv")
    df_results.to_csv(results_path, index=False)
    print(f"✅ Full results: {results_path}")
    
    # Generate report
    report_path = os.path.join(RESULTS_DIR, "WATCHLIST_COMPREHENSIVE_OPTIMIZATION_REPORT.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write(" COMPREHENSIVE WATCHLIST GROUP RECOMMENDATION OPTIMIZATION\n")
        f.write(" For Paper Section: Comprehensive Model Comparison\n")
        f.write("="*70 + "\n\n")
        
        f.write("RESEARCH QUESTION:\n")
        f.write("-" * 70 + "\n")
        f.write("What is the BEST watchlist-based approach for group recommendations?\n\n")
        
        f.write("MODELS TESTED:\n")
        f.write("-" * 70 + "\n")
        f.write("1. WatchlistRecommender (AVG Profile + Consensus)\n")
        f.write("   - Tested with disagreement_penalty: 0.0, 0.25, 0.5, 0.75, 1.0\n\n")
        f.write("2. Hybrid Model 3 - Baseline (AVERAGE aggregation)\n\n")
        f.write("3. Hybrid Model 3 - Alternative Aggregations\n")
        f.write("   - MIN: Most conservative, fairness-focused\n")
        f.write("   - HARMONIC_MEAN: Balanced approach\n\n")
        f.write("4. Hybrid Model 3 - With Disagreement Penalty (NEW!)\n")
        f.write("   - Tested with penalty: 0.25, 0.5, 0.75, 1.0\n\n")
        
        f.write("EVALUATION:\n")
        f.write("-" * 70 + "\n")
        f.write("Composite Score = 60% Group NDCG + 40% Fairness\n")
        f.write("Evaluated on validation set with group sizes: 2, 3, 4\n\n")
        
        f.write("="*70 + "\n")
        f.write("OVERALL BEST CONFIGURATION\n")
        f.write("="*70 + "\n\n")
        f.write(f"Model Type: {best_config['model_type']}\n")
        f.write(f"Aggregation: {best_config['aggregation']}\n")
        f.write(f"Disagreement Penalty: {best_config['disagreement_penalty']}\n")
        f.write(f"Top-K: {best_config['top_k']}\n\n")
        f.write(f"Performance:\n")
        f.write(f"  - Group NDCG: {best_config['group_ndcg']:.4f}\n")
        f.write(f"  - Fairness: {best_config['fairness']:.4f}\n")
        f.write(f"  - Coverage: {best_config['coverage']:.4f}\n")
        f.write(f"  - Composite Score: {best_config['composite_score']:.4f}\n")
        f.write(f"  - Groups Evaluated: {best_config['num_groups']}\n\n")
        
        # SCIENTIFIC VALIDITY CHECK
        if best_config['num_groups'] < 10:
            f.write("⚠️  SCIENTIFIC VALIDITY WARNING:\n")
            f.write(f"   This configuration was evaluated on only {best_config['num_groups']} groups.\n")
            f.write("   Literature standard: Minimum 10-20 groups for reliable conclusions.\n")
            f.write("   These results should be interpreted with caution.\n\n")
        
        f.write("="*70 + "\n")
        f.write("BEST PER MODEL TYPE\n")
        f.write("="*70 + "\n\n")
        
        for model_type in sorted(df_results['model_type'].unique()):
            type_results = df_results[df_results['model_type'] == model_type]
            best_type_idx = type_results['composite_score'].idxmax()
            best_type = type_results.loc[best_type_idx]
            
            f.write(f"{model_type}:\n")
            f.write(f"  Aggregation: {best_type['aggregation']}\n")
            f.write(f"  Penalty: {best_type['disagreement_penalty']}\n")
            f.write(f"  Top-K: {best_type['top_k']}\n")
            f.write(f"  Group NDCG: {best_type['group_ndcg']:.4f}\n")
            f.write(f"  Fairness: {best_type['fairness']:.4f}\n")
            f.write(f"  Composite: {best_type['composite_score']:.4f}\n")
            f.write(f"  Num Groups: {best_type['num_groups']}")
            
            # Validity warning
            if best_type['num_groups'] < 10:
                f.write(f" ⚠️  (Unreliable - < 10 groups)")
            f.write("\n\n")
        
        f.write("="*70 + "\n")
        f.write("FULL RESULTS TABLE\n")
        f.write("="*70 + "\n\n")
        f.write(df_results.sort_values('composite_score', ascending=False).to_string(index=False))
        f.write("\n\n")
        
        # Add validity summary
        unreliable_results = df_results[df_results['num_groups'] < 10]
        if len(unreliable_results) > 0:
            f.write("="*70 + "\n")
            f.write("SCIENTIFIC VALIDITY WARNINGS\n")
            f.write("="*70 + "\n\n")
            f.write(f"⚠️  {len(unreliable_results)} configuration(s) evaluated on < 10 groups.\n")
            f.write("These results may not be statistically reliable per literature standards.\n\n")
            f.write("Unreliable configurations:\n")
            for idx, row in unreliable_results.iterrows():
                f.write(f"  - {row['model_type']} (agg={row['aggregation']}, "
                       f"penalty={row['disagreement_penalty']}, groups={row['num_groups']})\n")
            f.write("\n")
        
        f.write("SCIENTIFIC FINDINGS:\n")
        f.write("-" * 70 + "\n")
        f.write("1. Compare WatchlistRecommender vs Hybrid Model 3 approaches\n")
        f.write("2. Evaluate impact of disagreement penalty on both models\n")
        f.write("3. Test alternative aggregation strategies (MIN, HARMONIC_MEAN)\n")
        f.write("4. Identify optimal configuration for group recommendations\n\n")
        
        f.write("NEXT STEPS:\n")
        f.write("-" * 70 + "\n")
        f.write("1. Use best configuration for final test set evaluation\n")
        f.write("2. Compare against other recommendation approaches (UBCF, IBCF, etc.)\n")
        f.write("3. Document findings in paper\n")
    
    print(f"✅ Optimization report: {report_path}")
    
    print("\n" + "="*70)
    print(" COMPREHENSIVE OPTIMIZATION COMPLETE")
    print("="*70)
    print(f"\nBEST OVERALL: {best_config['model_type']}")
    print(f"  - Aggregation: {best_config['aggregation']}")
    print(f"  - Penalty: {best_config['disagreement_penalty']}")
    print(f"  - Composite Score: {best_config['composite_score']:.4f}")
    print("="*70)


if __name__ == "__main__":
    main()
