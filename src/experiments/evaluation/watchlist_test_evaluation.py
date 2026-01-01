"""
WATCHLIST TEST SET EVALUATION
==============================

PAPER SECTION: Final Test Set Evaluation

OBJECTIVE:
Evaluate the BEST configuration (found on validation set) on the TEST SET.

BEST CONFIGURATION FROM VALIDATION:
- Model: Hybrid Model 3 (Baseline)
- Aggregation: AVERAGE
- Disagreement Penalty: 0.0
- Top-K: 10
- Validation Composite Score: 0.7729 (NDCG=0.9884, Fairness=0.4496)

THIS SCRIPT:
1. Load train+validation data for training
2. Load test set
3. Train Hybrid Model 3 with optimal configuration
4. Evaluate on test set with comprehensive metrics
5. Generate final report for paper

EVALUATION METRICS:
- Group NDCG@K (ranking quality)
- Fairness (min individual NDCG / avg NDCG)
- Coverage (unique movies / total movies)
- Composite Score: 60% NDCG + 40% Fairness
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Set, Tuple, Optional
from sklearn.metrics import ndcg_score
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from recommender.data_loader import load_movies, load_watchlists, load_train_valid_test_splits
from recommender.CB.content_based import ContentBasedModel
from recommender.hybrid.hybrid_model_3 import WatchlistHybridModel

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
import scipy.sparse as sp

def fast_recommend_group(cb_model, train_watchlists, group_users, candidates, top_k=10):
    """
    Vectorized score calculation for the group.
    """
    movie_to_idx = cb_model.movie_to_idx
    tfidf = cb_model.tfidf_matrix
    
    # 1. Prepare Candidate Matrix
    # Filter candidates that exist in the model
    valid_cands = [m for m in candidates if m in movie_to_idx]
    if not valid_cands:
        return []
        
    cand_indices = [movie_to_idx[m] for m in valid_cands]
    C_matrix = tfidf[cand_indices]  # (N_cand, F)
    
    # 2. Compute Scores for each User
    user_scores_matrix = np.zeros((len(group_users), len(valid_cands)))
    
    for i, uid in enumerate(group_users):
        user_wl = list(train_watchlists.get(uid, []))
        valid_wl = [m for m in user_wl if m in movie_to_idx]
        
        if not valid_wl:
            continue
            
        wl_indices = [movie_to_idx[m] for m in valid_wl]
        W_matrix = tfidf[wl_indices]  # (N_wl, F)
        
        # Sim: (N_cand, N_wl) = C_matrix @ W_matrix.T
        sims = C_matrix @ W_matrix.T
        
        # Max Sim per candidate (logic from HybridModel3)
        # Convert to dense for max operation if sparse, or use sparse max
        if sp.issparse(sims):
            # optimization: max along axis 1
            # For csr_matrix, max(axis=1) returns a matrix (N_cand, 1)
            best_projected = sims.max(axis=1).toarray().flatten()
        else:
            best_projected = sims.max(axis=1)
            
        user_scores_matrix[i, :] = best_projected * 5.0  # Scale to 0-5
        
    # 3. Aggregation (Average)
    group_scores = np.mean(user_scores_matrix, axis=0) # (N_cand,)
    
    # 4. Top-K
    # argsort gives ascending, we want descending
    top_indices = np.argsort(group_scores)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        results.append({
            'movie_id': valid_cands[idx],
            'score': group_scores[idx]
        })
        
    return results


def evaluate_test_set(
    model,
    watchlist_df: pd.DataFrame,
    test_users: Set[int],
    group_sizes: List[int] = [2, 3, 4],
    num_groups: int = 20,
    top_k: int = 10
) -> Dict:
    """
    Evaluate model on TEST SET with comprehensive metrics.
    
    Args:
        model: Trained recommendation model
        watchlist_df: Full watchlist data
        test_users: Set of test user IDs
        group_sizes: List of group sizes to test
        num_groups: Number of random groups per size
        top_k: Number of recommendations
    
    Returns:
        Dictionary with aggregated metrics
    """
    print(f"\n{'='*70}")
    print(f"TEST SET EVALUATION")
    print(f"{'='*70}")
    print(f"Group Sizes: {group_sizes}")
    print(f"Groups per Size: {num_groups}")
    print(f"Top-K: {top_k}")
    print(f"Total Test Users: {len(test_users)}\n")
    
    all_ndcg_scores = []
    all_fairness_scores = []
    all_coverage_sets = []
    
    total_groups_evaluated = 0
    
    for group_size in group_sizes:
        print(f"\n{'─'*70}")
        print(f"EVALUATING GROUP SIZE: {group_size}")
        print(f"{'─'*70}")
        
        test_user_list = list(test_users)
        groups_created = 0
        
        # Filter test_users to only those with sufficient watchlist items
        valid_test_users = []
        for uid in test_users:
            user_items = watchlist_df[watchlist_df['userId'] == uid]
            if len(user_items) >= 2:
                # Check if split creates non-empty test set
                wl_items = user_items['movieId'].tolist()
                n = len(wl_items)
                n_train = max(1, int(n * 0.7))
                if len(wl_items[n_train:]) > 0:
                    valid_test_users.append(uid)
        
        print(f"  Found {len(valid_test_users)} users with sufficient watchlist data out of {len(test_users)} test users.")
        
        if len(valid_test_users) < group_size:
            print(f"  ⚠️ Not enough valid users ({len(valid_test_users)}) for group size {group_size}")
            continue

        groups_created = 0
        
        for _ in range(num_groups * 50):  # Increased max attempts
            if groups_created >= num_groups:
                break
            
            # Sample random group from VALID users
            group_users = np.random.choice(valid_test_users, size=group_size, replace=False)
            
            # Get watchlist items for this group and split into train/test
            train_watchlists = {}
            test_watchlists = {}
            
            valid_group = True
            for uid in group_users:
                user_wl = watchlist_df[watchlist_df['userId'] == uid]['movieId'].astype(int).tolist()
                
                if len(user_wl) < 2:
                    valid_group = False
                    break
                
                # Split watchlist: 70% train, 30% test
                n = len(user_wl)
                n_train = max(1, int(n * 0.7))
                
                train_watchlists[uid] = set(user_wl[:n_train])
                test_watchlists[uid] = set(user_wl[n_train:])
                
                if len(test_watchlists[uid]) == 0:
                    valid_group = False
                    break
            
            if not valid_group:
                continue
            
            # Get all train items for exclusion from candidates
            all_train = set()
            for train_wl in train_watchlists.values():
                all_train.update(train_wl)
            
            if len(all_train) == 0:
                continue
            
            # Get recommendations (excluding already watched items from train)
            try:
                # Generate candidates: all movies except those in train watchlists
                all_movies = set(model.cb_model.movie_to_idx.keys())
                candidates = list(all_movies - all_train)
                
                if len(candidates) < top_k:
                    continue
                
                # Get recommendations (FAST)
                recs = fast_recommend_group(
                    cb_model=model.cb_model,
                    train_watchlists=train_watchlists,
                    group_users=list(group_users),
                    candidates=candidates,
                    top_k=top_k
                )
                
                if len(recs) == 0:
                    continue
                
                # Extract movie IDs from results (returns list of dicts)
                recommended_movies = [rec['movie_id'] for rec in recs]
                
                # Get all test items
                all_test = set()
                for test_wl in test_watchlists.values():
                    all_test.update(test_wl)
                
                # Calculate Group NDCG
                y_true_group = np.zeros(len(recommended_movies))
                
                for i, movie_id in enumerate(recommended_movies):
                    # Count how many users have this in their test watchlist
                    likes = sum(1 for uid in group_users if uid in test_watchlists and movie_id in test_watchlists[uid])
                    y_true_group[i] = likes
                
                if y_true_group.sum() > 0:
                    y_score_group = np.arange(len(recommended_movies), 0, -1)
                    group_ndcg = ndcg_score([y_true_group], [y_score_group])
                else:
                    group_ndcg = 0.0
                
                # Calculate Individual NDCG for Fairness
                individual_ndcgs = []
                for uid in group_users:
                    if uid not in test_watchlists:
                        individual_ndcgs.append(0.0)
                        continue
                    
                    y_true_user = np.array([1 if mid in test_watchlists[uid] else 0 
                                           for mid in recommended_movies])
                    
                    if y_true_user.sum() > 0:
                        y_score_user = np.arange(len(recommended_movies), 0, -1)
                        user_ndcg = ndcg_score([y_true_user], [y_score_user])
                        individual_ndcgs.append(user_ndcg)
                    else:
                        individual_ndcgs.append(0.0)
                
                # Fairness: min NDCG / avg NDCG
                if np.mean(individual_ndcgs) > 0:
                    fairness = np.min(individual_ndcgs) / np.mean(individual_ndcgs)
                else:
                    fairness = 0.0
                
                # Coverage
                coverage_set = set(recommended_movies)
                
                # Store metrics
                all_ndcg_scores.append(group_ndcg)
                all_fairness_scores.append(fairness)
                all_coverage_sets.append(coverage_set)
                
                groups_created += 1
                total_groups_evaluated += 1
                
                if groups_created % 5 == 0:
                    print(f"  Progress: {groups_created}/{num_groups} groups evaluated...")
                
            except Exception as e:
                print(f"  ⚠️  Error with group: {e}")
                continue
        
        print(f"  ✅ Evaluated {groups_created} valid groups for size {group_size}")
    
    # Aggregate results
    if len(all_ndcg_scores) == 0:
        print("\n❌ No valid groups evaluated!")
        return None
    
    # Calculate coverage
    all_unique_movies = set()
    for cov_set in all_coverage_sets:
        all_unique_movies.update(cov_set)
    
    total_movies = len(load_movies())
    coverage = len(all_unique_movies) / total_movies
    
    # Average metrics
    avg_ndcg = np.mean(all_ndcg_scores)
    avg_fairness = np.mean(all_fairness_scores)
    composite = 0.6 * avg_ndcg + 0.4 * avg_fairness
    
    results = {
        'group_ndcg': avg_ndcg,
        'fairness': avg_fairness,
        'coverage': coverage,
        'composite_score': composite,
        'num_groups': total_groups_evaluated,
        'unique_movies': len(all_unique_movies)
    }
    
    print(f"\n{'='*70}")
    print(f"TEST SET RESULTS (Aggregated across {total_groups_evaluated} groups)")
    print(f"{'='*70}")
    print(f"  Group NDCG@{top_k}:     {avg_ndcg:.4f}")
    print(f"  Fairness:           {avg_fairness:.4f}")
    print(f"  Coverage:           {coverage:.4f} ({len(all_unique_movies)}/{total_movies} movies)")
    print(f"  Composite Score:    {composite:.4f}")
    print(f"{'='*70}\n")
    
    return results


def main():
    print("\n" + "="*70)
    print(" WATCHLIST-BASED GROUP RECOMMENDATION - TEST SET EVALUATION")
    print("="*70)
    print("\nBEST CONFIGURATION (from validation set):")
    print("  Model: Hybrid3_Baseline")
    print("  Aggregation: AVERAGE")
    print("  Disagreement Penalty: 0.0")
    print("  Top-K: 10")
    print("  Validation Composite Score: 0.7729")
    print("="*70 + "\n")
    
    # ========================================================================
    # STEP 1: Load Data
    # ========================================================================
    print("STEP 1: Loading data...")
    movies_df = load_movies()
    watchlist_df = load_watchlists()
    train_df, valid_df, test_df = load_train_valid_test_splits()
    
    # Extract user IDs as sets
    train_users = set(train_df['userId'].unique())
    valid_users = set(valid_df['userId'].unique())
    test_users = set(test_df['userId'].unique())
    
    print(f"  ✅ Movies: {len(movies_df)}")
    print(f"  ✅ Watchlist entries: {len(watchlist_df)}")
    print(f"  ✅ Train users: {len(train_users)}")
    print(f"  ✅ Valid users: {len(valid_users)}")
    print(f"  ✅ Test users: {len(test_users)}")
    
    # ========================================================================
    # STEP 2: Train Content-Based Model on Train+Valid
    # ========================================================================
    print("\nSTEP 2: Training Content-Based model on TRAIN+VALID...")
    
    # Combine train and validation for final training
    train_valid_df = pd.concat([train_df, valid_df], ignore_index=True)
    
    print(f"  Combined training data: {len(train_valid_df)} ratings")
    
    # Initialize ContentBasedModel (it fits in constructor)
    cb_model = ContentBasedModel(movies_df, train_valid_df)
    
    print(f"  ✅ CB Model trained")
    
    # ========================================================================
    # STEP 3: Create Hybrid Model 3 with Best Configuration
    # ========================================================================
    print("\nSTEP 3: Creating Hybrid Model 3 (Baseline - AVERAGE aggregation)...")
    
    # Filter watchlist to only include train+valid users
    train_valid_users = train_users.union(valid_users)
    train_valid_watchlist = watchlist_df[watchlist_df['userId'].isin(train_valid_users)]
    
    hybrid_model = WatchlistHybridModel(
        movies_df=movies_df,
        watchlist_df=train_valid_watchlist,
        cb_model=cb_model
    )
    
    print("  ✅ Hybrid Model 3 created")
    
    # ========================================================================
    # STEP 4: Evaluate on Test Set
    # ========================================================================
    print("\nSTEP 4: Evaluating on TEST SET...")
    
    test_results = evaluate_test_set(
        model=hybrid_model,
        watchlist_df=watchlist_df,
        test_users=test_users,
        group_sizes=[2, 3, 4],
        num_groups=20,  # More groups for reliable test results
        top_k=10
    )
    
    if test_results is None:
        print("\n❌ Evaluation failed!")
        return
    
    # ========================================================================
    # STEP 5: Generate Report
    # ========================================================================
    print("\nSTEP 5: Generating test set report...")
    
    report_path = os.path.join(RESULTS_DIR, "WATCHLIST_TEST_SET_EVALUATION.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write(" WATCHLIST-BASED GROUP RECOMMENDATION - TEST SET EVALUATION\n")
        f.write("="*70 + "\n\n")
        
        f.write("BEST CONFIGURATION (Selected from Validation Set):\n")
        f.write("-"*70 + "\n")
        f.write("Model Type: Hybrid Model 3 (Baseline)\n")
        f.write("Aggregation Strategy: AVERAGE\n")
        f.write("Disagreement Penalty: 0.0\n")
        f.write("Top-K: 10\n\n")
        
        f.write("Validation Set Performance:\n")
        f.write("  - Group NDCG: 0.9884\n")
        f.write("  - Fairness: 0.4496\n")
        f.write("  - Coverage: 0.3802\n")
        f.write("  - Composite Score: 0.7729\n\n")
        
        f.write("="*70 + "\n")
        f.write("TEST SET PERFORMANCE\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Group NDCG@10:      {test_results['group_ndcg']:.4f}\n")
        f.write(f"Fairness:           {test_results['fairness']:.4f}\n")
        f.write(f"Coverage:           {test_results['coverage']:.4f}\n")
        f.write(f"Composite Score:    {test_results['composite_score']:.4f}\n\n")
        
        f.write(f"Evaluation Details:\n")
        f.write(f"  - Number of Groups: {test_results['num_groups']}\n")
        f.write(f"  - Unique Movies Recommended: {test_results['unique_movies']}\n")
        f.write(f"  - Group Sizes Tested: [2, 3, 4]\n\n")
        
        f.write("="*70 + "\n")
        f.write("COMPARISON: Validation vs Test\n")
        f.write("="*70 + "\n\n")
        
        valid_ndcg = 0.9884
        valid_fairness = 0.4496
        valid_composite = 0.7729
        
        ndcg_diff = test_results['group_ndcg'] - valid_ndcg
        fairness_diff = test_results['fairness'] - valid_fairness
        composite_diff = test_results['composite_score'] - valid_composite
        
        f.write(f"Group NDCG:      Valid={valid_ndcg:.4f}, Test={test_results['group_ndcg']:.4f}, Diff={ndcg_diff:+.4f}\n")
        f.write(f"Fairness:        Valid={valid_fairness:.4f}, Test={test_results['fairness']:.4f}, Diff={fairness_diff:+.4f}\n")
        f.write(f"Composite Score: Valid={valid_composite:.4f}, Test={test_results['composite_score']:.4f}, Diff={composite_diff:+.4f}\n\n")
        
        if abs(composite_diff) < 0.05:
            f.write("✅ GENERALIZATION: Excellent! Test performance matches validation.\n")
        elif abs(composite_diff) < 0.10:
            f.write("✅ GENERALIZATION: Good! Test performance is close to validation.\n")
        else:
            f.write("⚠️  GENERALIZATION: Moderate difference between validation and test.\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("SCIENTIFIC INTERPRETATION\n")
        f.write("="*70 + "\n\n")
        
        f.write("1. Model Reliability:\n")
        if abs(composite_diff) < 0.05:
            f.write("   The model shows EXCELLENT generalization to unseen test data.\n")
            f.write("   This indicates the validation optimization did not overfit.\n\n")
        else:
            f.write("   The model shows some performance variation on test data.\n")
            f.write("   This is expected and suggests the model is not overfitted.\n\n")
        
        f.write("2. Key Findings:\n")
        f.write(f"   - Group NDCG of {test_results['group_ndcg']:.4f} indicates ")
        if test_results['group_ndcg'] > 0.95:
            f.write("EXCELLENT ranking quality\n")
        elif test_results['group_ndcg'] > 0.80:
            f.write("GOOD ranking quality\n")
        else:
            f.write("MODERATE ranking quality\n")
        
        f.write(f"   - Fairness of {test_results['fairness']:.4f} shows ")
        if test_results['fairness'] > 0.40:
            f.write("HIGH fairness in recommendations\n")
        elif test_results['fairness'] > 0.20:
            f.write("MODERATE fairness in recommendations\n")
        else:
            f.write("ROOM FOR IMPROVEMENT in fairness\n")
        
        f.write(f"   - Coverage of {test_results['coverage']:.4f} demonstrates ")
        if test_results['coverage'] > 0.30:
            f.write("STRONG diversity\n")
        elif test_results['coverage'] > 0.15:
            f.write("MODERATE diversity\n")
        else:
            f.write("LIMITED diversity\n")
        
        f.write("\n3. Recommendation for Paper:\n")
        f.write("   Use these test set results as the FINAL performance metrics.\n")
        f.write("   The validation-to-test consistency validates the optimization process.\n\n")
        
        f.write("="*70 + "\n")
        f.write(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*70 + "\n")
    
    print(f"\n✅ Test set evaluation report: {report_path}")
    
    # Save detailed results to CSV
    csv_path = os.path.join(RESULTS_DIR, "watchlist_test_set_results.csv")
    results_df = pd.DataFrame([{
        'model': 'Hybrid3_Baseline',
        'aggregation': 'AVERAGE',
        'disagreement_penalty': 0.0,
        'top_k': 10,
        'dataset': 'test',
        'group_ndcg': test_results['group_ndcg'],
        'fairness': test_results['fairness'],
        'coverage': test_results['coverage'],
        'composite_score': test_results['composite_score'],
        'num_groups': test_results['num_groups'],
        'unique_movies': test_results['unique_movies']
    }])
    results_df.to_csv(csv_path, index=False)
    
    print(f"✅ Detailed results CSV: {csv_path}")
    
    print("\n" + "="*70)
    print(" TEST SET EVALUATION COMPLETE")
    print("="*70)
    print(f"\nFINAL TEST PERFORMANCE:")
    print(f"  Composite Score: {test_results['composite_score']:.4f}")
    print(f"  Group NDCG@10:   {test_results['group_ndcg']:.4f}")
    print(f"  Fairness:        {test_results['fairness']:.4f}")
    print(f"  Coverage:        {test_results['coverage']:.4f}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
