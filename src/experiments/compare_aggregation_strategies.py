"""
GROUP AGGREGATION STRATEGIES COMPARISON
========================================

RESEARCH QUESTION: Which aggregation strategy performs best for watchlist-based group recommendations?

SCIENTIFIC BACKGROUND:
- Masthoff (2011): "Group Recommender Systems: Combining Individual Models"
- Amer-Yahia et al. (2009): "Getting Recommendations for Groups"

STRATEGIES COMPARED:
1. Average (AVG) - Baseline
2. Least Misery (LM) - Conservative
3. Most Pleasure (MP) - Optimistic  
4. Hybrid 70/30 (H70) - Balanced
5. Hybrid 50/50 (H50) - Equal weight

EXPECTED RESULTS:
- LM: Highest Fairness, Lowest NDCG
- MP: Lowest Fairness, Highest NDCG
- AVG: Baseline
- Hybrid: Best trade-off
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Callable
from sklearn.metrics import ndcg_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from recommender.data_loader import load_movies, load_watchlists, load_train_valid_test_splits
from recommender.CB.content_based import ContentBasedModel
from recommender.hybrid.hybrid_model_3 import WatchlistHybridModel

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


# ===== AGGREGATION STRATEGIES =====

def average_aggregation(scores: List[float]) -> float:
    """
    Average Strategy (Masthoff, 2004)
    - Treats all members equally
    - Standard baseline
    """
    return np.mean(scores) if scores else 0.0


def least_misery_aggregation(scores: List[float]) -> float:
    """
    Least Misery Strategy (Amer-Yahia et al., 2009)
    - Ensures no member is very dissatisfied
    - Conservative approach
    - Good for close groups (families, couples)
    """
    return np.min(scores) if scores else 0.0


def most_pleasure_aggregation(scores: List[float]) -> float:
    """
    Most Pleasure Strategy
    - Ensures at least one member is very satisfied
    - Optimistic approach
    - Good for leader-follower groups
    """
    return np.max(scores) if scores else 0.0


def hybrid_aggregation(scores: List[float], alpha: float = 0.7) -> float:
    """
    Hybrid Strategy (Masthoff, 2011)
    - Combines average and least misery
    - alpha: weight for average (0-1)
    - (1-alpha): weight for least misery
    
    alpha=0.7: 70% average, 30% least misery (recommended)
    alpha=0.5: 50% average, 50% least misery (balanced)
    """
    if not scores:
        return 0.0
    return alpha * np.mean(scores) + (1 - alpha) * np.min(scores)


# ===== EVALUATION FUNCTION =====

def evaluate_group_with_strategy(
    group: List[int],
    model,
    all_ratings: pd.DataFrame,
    watchlist: pd.DataFrame,
    movies: pd.DataFrame,
    aggregation_func: Callable,
    rating_threshold: float = 3.5
) -> Dict[str, float]:
    """
    Evaluate a single group with a specific aggregation strategy.
    """
    try:
        # 1. Ground Truth
        group_ratings = all_ratings[all_ratings['userId'].isin(group)]
        group_high_rated = group_ratings[group_ratings['rating'] >= rating_threshold]
        
        min_support = max(1, int(len(group) * 0.5))
        movie_counts = group_high_rated.groupby('movieId').size()
        ground_truth_movies = set(movie_counts[movie_counts >= min_support].index)
        
        if len(ground_truth_movies) == 0:
            return None
        
        # 2. Get individual recommendations for each member
        all_recs = {}
        for uid in group:
            user_watchlist = set(watchlist[watchlist['userId'] == uid]['movieId'].tolist())
            if len(user_watchlist) == 0:
                continue
            
            all_movies = set(model.cb_model.movie_to_idx.keys())
            candidates = all_movies - user_watchlist
            
            user_recs = {}
            for mid in candidates:
                try:
                    score = model.predict(uid, mid)
                    if not np.isnan(score):
                        user_recs[mid] = score
                except:
                    continue
            
            all_recs[uid] = user_recs
        
        if len(all_recs) == 0:
            return None
        
        # 3. Aggregate using the specified strategy
        all_candidate_movies = set()
        for user_recs in all_recs.values():
            all_candidate_movies.update(user_recs.keys())
        
        aggregated = []
        for mid in all_candidate_movies:
            scores = [all_recs[uid].get(mid, 0.0) for uid in all_recs.keys()]
            # Only aggregate if at least one member has a score
            valid_scores = [s for s in scores if s > 0]
            if valid_scores:
                group_score = aggregation_func(valid_scores)
                aggregated.append((mid, group_score))
        
        if len(aggregated) == 0:
            return None
        
        # Sort and get top-10
        aggregated = sorted(aggregated, key=lambda x: x[1], reverse=True)[:10]
        rec_ids = [mid for mid, _ in aggregated]
        rec_scores = [score for _, score in aggregated]
        
        # 4. Calculate Metrics
        
        # NDCG
        y_true = [1 if mid in ground_truth_movies else 0 for mid in rec_ids]
        if sum(y_true) > 0:
            ndcg = ndcg_score([y_true], [rec_scores])
        else:
            ndcg = 0.0
        
        # Precision & Recall
        hits = len(set(rec_ids) & ground_truth_movies)
        precision = hits / len(rec_ids) if len(rec_ids) > 0 else 0.0
        recall = hits / len(ground_truth_movies) if len(ground_truth_movies) > 0 else 0.0
        
        # Fairness: Individual Satisfaction
        members_satisfied = 0
        for uid in group:
            user_satisfied = False
            for mid in rec_ids[:5]:  # Top-5
                try:
                    score = model.predict(uid, mid)
                    if not np.isnan(score) and score >= 3.5:
                        user_satisfied = True
                        break
                except:
                    continue
            if user_satisfied:
                members_satisfied += 1
        
        fairness = members_satisfied / len(group) if len(group) > 0 else 0.0
        
        # Disagreement: Std of scores for recommended movies
        disagreements = []
        for mid in rec_ids:
            scores = [all_recs[uid].get(mid, 0.0) for uid in all_recs.keys()]
            valid_scores = [s for s in scores if s > 0]
            if len(valid_scores) > 1:
                disagreements.append(np.std(valid_scores))
        
        avg_disagreement = np.mean(disagreements) if disagreements else 0.0
        
        return {
            'ndcg': ndcg,
            'precision': precision,
            'recall': recall,
            'fairness': fairness,
            'disagreement': avg_disagreement,
            'ground_truth_size': len(ground_truth_movies)
        }
        
    except Exception as e:
        print(f"Error: {e}")
        return None


# ===== MAIN EXPERIMENT =====

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print("="*70)
    print(" GROUP AGGREGATION STRATEGIES COMPARISON")
    print(" Scientific Evaluation of Aggregation Methods")
    print("="*70)
    
    # Load data
    print("\n[1] Loading Data...")
    movies = load_movies()
    watchlist = load_watchlists()
    train_df, valid_df, test_df = load_train_valid_test_splits()
    all_ratings = pd.concat([train_df, valid_df, test_df], ignore_index=True)
    
    print(f"  Movies: {len(movies)}")
    print(f"  Watchlist entries: {len(watchlist)}")
    print(f"  Total ratings: {len(all_ratings)}")
    
    # Build model
    print("\n[2] Building Content-Based Model...")
    cbf = ContentBasedModel(movies, train_df)
    model = WatchlistHybridModel(movies, watchlist, cbf)
    
    # Get evaluation users
    watchlist_users = set(watchlist['userId'].unique())
    ratings_users = set(all_ratings['userId'].unique())
    eval_users_list = list(watchlist_users & ratings_users)
    
    print(f"\n[3] Users with both watchlist and ratings: {len(eval_users_list)}")
    
    # Define strategies
    strategies = {
        'Average (AVG)': average_aggregation,
        'Least Misery (LM)': least_misery_aggregation,
        'Most Pleasure (MP)': most_pleasure_aggregation,
        'Hybrid 70/30 (H70)': lambda scores: hybrid_aggregation(scores, alpha=0.7),
        'Hybrid 50/50 (H50)': lambda scores: hybrid_aggregation(scores, alpha=0.5),
    }
    
    # Evaluate all strategies
    print("\n" + "="*70)
    print("EVALUATING AGGREGATION STRATEGIES")
    print("="*70)
    
    all_results = []
    
    for group_size in [2, 3, 4]:
        print(f"\n{'#'*70}")
        print(f"GROUP SIZE: {group_size}")
        print(f"{'#'*70}")
        
        num_groups = min(5, len(eval_users_list) // group_size)
        if num_groups < 1:
            print(f"Not enough users for groups of size {group_size}")
            continue
        
        for strategy_name, strategy_func in strategies.items():
            print(f"\n--- Strategy: {strategy_name} ---")
            
            metrics = {
                'ndcg': [],
                'precision': [],
                'recall': [],
                'fairness': [],
                'disagreement': [],
                'ground_truth_size': []
            }
            
            for i in range(num_groups):
                start_idx = i * group_size
                group = eval_users_list[start_idx:start_idx + group_size]
                
                result = evaluate_group_with_strategy(
                    group=group,
                    model=model,
                    all_ratings=all_ratings,
                    watchlist=watchlist,
                    movies=movies,
                    aggregation_func=strategy_func
                )
                
                if result:
                    for key, value in result.items():
                        metrics[key].append(value)
            
            if metrics['ndcg']:
                avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
                
                all_results.append({
                    'Strategy': strategy_name,
                    'Group_Size': group_size,
                    'NDCG@10': avg_metrics['ndcg'],
                    'Precision@10': avg_metrics['precision'],
                    'Recall@10': avg_metrics['recall'],
                    'Fairness': avg_metrics['fairness'],
                    'Disagreement': avg_metrics['disagreement'],
                    'Avg_GT_Size': avg_metrics['ground_truth_size'],
                    'Num_Groups': len(metrics['ndcg'])
                })
                
                print(f"  Evaluated: {len(metrics['ndcg'])} groups")
                print(f"  NDCG@10:       {avg_metrics['ndcg']:.4f}")
                print(f"  Precision@10:  {avg_metrics['precision']:.4f}")
                print(f"  Recall@10:     {avg_metrics['recall']:.4f}")
                print(f"  Fairness:      {avg_metrics['fairness']:.4f}")
                print(f"  Disagreement:  {avg_metrics['disagreement']:.4f}")
            else:
                print(f"  No valid groups evaluated")
    
    # Save results
    df_results = pd.DataFrame(all_results)
    
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print("\n" + df_results.to_string(index=False))
    
    results_path = os.path.join(RESULTS_DIR, "aggregation_strategies_comparison.csv")
    df_results.to_csv(results_path, index=False)
    
    print(f"\n✅ Results saved to: {results_path}")
    
    # Analysis
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)
    
    for size in [2, 3, 4]:
        size_results = df_results[df_results['Group_Size'] == size]
        if len(size_results) > 0:
            print(f"\n--- Group Size {size} ---")
            best_ndcg = size_results.loc[size_results['NDCG@10'].idxmax()]
            best_fairness = size_results.loc[size_results['Fairness'].idxmax()]
            
            print(f"Best NDCG:     {best_ndcg['Strategy']} ({best_ndcg['NDCG@10']:.4f})")
            print(f"Best Fairness: {best_fairness['Strategy']} ({best_fairness['Fairness']:.4f})")
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
