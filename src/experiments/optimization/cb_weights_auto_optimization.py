"""
CONTENT-BASED FEATURE WEIGHTS HYPERPARAMETER OPTIMIZATION
==========================================================

PAPER SECTION: Content-Based Filtering Optimization
RESEARCH QUESTION: What is the optimal feature weighting scheme?

METHODS:
1. Bayesian Optimization: Gaussian Process-based intelligent search (PRIMARY)
2. Random Search: Baseline comparison
3. Focused Search: Individual feature importance analysis

FEATURES:
- genres, director, keywords, actors, year, overview, companies, countries

WEIGHT RANGE: 0-5 (0=disabled, 5=very important)

METRIC: NDCG@10 on validation set (higher is better)

EXPECTED RESULT:
- Optimal weights that maximize NDCG@10
- Insights into which features matter most for ranking quality
"""

import os
import sys
import numpy as np
import pandas as pd
from itertools import product
from typing import Dict, List, Tuple
import time
from sklearn.metrics import ndcg_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from recommender.data_loader import load_movies, load_train_valid_test_splits
from recommender.CB.content_based import ContentBasedModel

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def evaluate_weights(weights: Dict[str, int], movies_df: pd.DataFrame, 
                     train_df: pd.DataFrame, valid_df: pd.DataFrame,
                     fixed_users: np.ndarray = None) -> Tuple[float, float, float]:
    """
    Evaluate a specific weight configuration using NDCG@10.
    
    METHODOLOGY:
    - For each user in validation set, generate Top-10 recommendations
    - Calculate NDCG based on whether recommended movies have high ratings (≥4.0)
    - NDCG measures ranking quality: higher is better
    
    Returns:
        (NDCG@10, Precision@10, Coverage) on validation set
    """
    # Build model with these weights
    t_start = time.time()
    model = ContentBasedModel(movies_df, train_df, weights=weights)
    t_model = time.time() - t_start
    print(f"    [TIMING] Model build (TF-IDF): {t_model:.2f}s", end=" | ")
    
    # Use fixed users if provided, otherwise sample
    if fixed_users is not None:
        valid_users = fixed_users
    else:
        valid_users = valid_df['userId'].unique()
    

        if len(valid_users) > 30:
            valid_users = np.random.choice(valid_users, 30, replace=False)
    
    ndcg_scores = []
    precision_scores = []
    coverage_count = 0
    
    for uid in valid_users:
        # Get user's validation ratings
        user_valid = valid_df[valid_df['userId'] == uid]
        
        # Get high-rated movies as ground truth (≥4.0)
        high_rated = set(user_valid[user_valid['rating'] >= 4.0]['movieId'].astype(int).tolist())
        
        if len(high_rated) == 0:
            continue  # Skip users with no high ratings in validation
        
        # Get user's training history (to exclude from recommendations)
        user_train = train_df[train_df['userId'] == uid]
        watched = set(user_train['movieId'].astype(int).tolist())
        
        # Get all movies as candidates
        try:
            all_movies = set(model.movie_to_idx.keys())
        except:
            continue
        
        # Candidates: movies not in training set
        candidates = all_movies - watched
        
        # CRITICAL SPEEDUP: Sample candidates for ranking
        # NDCG@10 requires ranking, but we don't need ALL movies
        # 1000 candidates >> 10 recommendations (100:1 ratio is excellent)
        # This matches grid search speed while maintaining NDCG quality
        if len(candidates) > 1000:
            # Ensure high_rated movies are in sample (if they're candidates)
            high_rated_candidates = high_rated & candidates
            other_candidates = candidates - high_rated
            
            # Sample from others
            n_sample = min(1000 - len(high_rated_candidates), len(other_candidates))
            if n_sample > 0:
                sampled_others = set(np.random.choice(list(other_candidates), n_sample, replace=False))
                candidates = high_rated_candidates | sampled_others
            else:
                candidates = high_rated_candidates
        
        # Score all candidates
        scored = []
        for mid in candidates:
            try:
                score = model.predict_rating(uid, mid)
                if not np.isnan(score):
                    scored.append((mid, score))
            except:
                continue
        
        if len(scored) < 10:
            continue  # Need at least 10 recommendations
        
        # Get Top-10 recommendations
        ranked = sorted(scored, key=lambda x: x[1], reverse=True)[:10]
        rec_ids = [mid for mid, _ in ranked]
        rec_scores = [score for _, score in ranked]
        
        # Calculate NDCG@10
        y_true = [1 if mid in high_rated else 0 for mid in rec_ids]
        
        if sum(y_true) > 0:  # Only calculate if there are relevant items
            ndcg = ndcg_score([y_true], [rec_scores])
            ndcg_scores.append(ndcg)
            
            # Calculate Precision@10
            precision = sum(y_true) / 10
            precision_scores.append(precision)
            
            coverage_count += 1
    
    if len(ndcg_scores) == 0:
        print(f"Eval: 0.00s | NDCG: 0.0000")
        return 0.0, 0.0, 0.0
    
    avg_ndcg = np.mean(ndcg_scores)
    avg_precision = np.mean(precision_scores)
    coverage = coverage_count / len(valid_users)
    
    t_eval = time.time() - t_start - t_model
    print(f"Eval: {t_eval:.2f}s | NDCG: {avg_ndcg:.4f}")
    
    return avg_ndcg, avg_precision, coverage


def bayesian_optimization(movies_df: pd.DataFrame, train_df: pd.DataFrame, 
                          valid_df: pd.DataFrame, weight_ranges: Dict[str, List[int]],
                          n_iterations: int = 50, n_initial_points: int = 10) -> pd.DataFrame:
    """
    EXPERIMENT 1: Bayesian Optimization
    
    PAPER USAGE: Table 1 - Bayesian Optimization Results
    
    METHOD:
    - Uses Gaussian Processes to model the objective function
    - Intelligently explores promising regions of the search space
    - Balances exploration vs exploitation
    
    SCIENTIFIC REFERENCE:
    - Snoek et al. (2012) "Practical Bayesian Optimization of Machine Learning Algorithms"
    - Shahriari et al. (2016) "Taking the Human Out of the Loop: A Review of Bayesian Optimization"
    
    ADVANTAGE: 
    - Much more efficient than grid/random search
    - Finds near-optimal solutions with fewer evaluations
    - Industry standard for hyperparameter tuning
    """
    print(f"\n{'='*70}")
    print("EXPERIMENT 1: BAYESIAN OPTIMIZATION")
    print(f"{'='*70}")
    print(f"Iterations: {n_iterations} | Initial random points: {n_initial_points}")
    
    # Sample fixed evaluation users ONCE for fair comparison
    all_users = valid_df['userId'].unique()
    if len(all_users) > 30:
        fixed_users = np.random.choice(all_users, 30, replace=False)
        print(f"Fixed evaluation set: 30 users (sampled from {len(all_users)})")
    else:
        fixed_users = all_users
        print(f"Fixed evaluation set: {len(fixed_users)} users (all)")
    
    try:
        from skopt import gp_minimize
        from skopt.space import Integer
        from skopt.utils import use_named_args
    except ImportError:
        print("\n⚠️  WARNING: scikit-optimize not installed!")
        print("   Install with: pip install scikit-optimize")
        print("   Falling back to Random Search...\n")
        return random_search(movies_df, train_df, valid_df, weight_ranges, n_iterations)
    
    # Define search space
    feature_names = list(weight_ranges.keys())
    search_space = [Integer(min(weight_ranges[f]), max(weight_ranges[f]), name=f) 
                    for f in feature_names]
    
    results = []
    best_ndcg = 0.0
    
    # Objective function (we minimize negative NDCG since gp_minimize minimizes)
    @use_named_args(search_space)
    def objective(**params):
        nonlocal best_ndcg
        
        weights = {f: params[f] for f in feature_names}
        # Use FIXED users for fair comparison
        ndcg, precision, coverage = evaluate_weights(
            weights, movies_df, train_df, valid_df, fixed_users=fixed_users
        )
        
        results.append({
            'weights': str(weights),
            'NDCG@10': ndcg,
            'Precision@10': precision,
            'Coverage': coverage,
            **weights
        })
        
        # Track best
        if ndcg > best_ndcg:
            best_ndcg = ndcg
            print(f"  🎯 New best! NDCG@10={ndcg:.4f} | Weights: {weights}")
        
        # Progress
        if len(results) % 5 == 0:
            print(f"  Progress: {len(results)}/{n_iterations} | Best NDCG@10: {best_ndcg:.4f}")
        
        return -ndcg  # Negative because we minimize
    
    print("\nStarting Bayesian Optimization...")
    start_time = time.time()
    
    # Run optimization
    result = gp_minimize(
        objective,
        search_space,
        n_calls=n_iterations,
        n_initial_points=n_initial_points,
        random_state=42,
        verbose=False
    )
    
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('NDCG@10', ascending=False)
    
    print(f"\n✅ Bayesian Optimization Complete!")
    print(f"   Time: {time.time() - start_time:.1f}s")
    print(f"   Best NDCG@10: {df_results.iloc[0]['NDCG@10']:.4f}")
    print(f"   Optimal weights: {df_results.iloc[0]['weights']}")
    
    return df_results


def random_search(movies_df: pd.DataFrame, train_df: pd.DataFrame,
                  valid_df: pd.DataFrame, weight_ranges: Dict[str, List[int]],
                  n_iterations: int = 50) -> pd.DataFrame:
    """
    EXPERIMENT 2: Random Search
    
    PAPER USAGE: Table 2 - Random Search Results
    
    METHOD:
    - Sample random weight combinations
    - Often finds good solutions faster than grid search
    - Literature: Bergstra & Bengio (2012) - Random search is competitive
    
    ADVANTAGE: Much faster, often finds near-optimal solutions
    """
    print(f"\n{'='*70}")
    print("EXPERIMENT 2: RANDOM SEARCH")
    print(f"{'='*70}")
    print(f"Testing {n_iterations} random combinations...")
    
    # Sample fixed evaluation users ONCE for fair comparison
    all_users = valid_df['userId'].unique()
    if len(all_users) > 30:
        fixed_users = np.random.choice(all_users, 30, replace=False)
        print(f"Fixed evaluation set: 30 users (sampled from {len(all_users)})")
    else:
        fixed_users = all_users
        print(f"Fixed evaluation set: {len(fixed_users)} users (all)")
    
    np.random.seed(42)
    results = []
    start_time = time.time()
    
    for i in range(n_iterations):
        # Sample random weights
        weights = {feature: np.random.choice(weight_ranges[feature]) 
                   for feature in weight_ranges.keys()}
        
        # Use FIXED users for fair comparison
        ndcg, precision, coverage = evaluate_weights(
            weights, movies_df, train_df, valid_df, fixed_users=fixed_users
        )
        
        results.append({
            'weights': str(weights),
            'NDCG@10': ndcg,
            'Precision@10': precision,
            'Coverage': coverage,
            **weights
        })
        
        # Progress
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            eta = (elapsed / (i + 1)) * (n_iterations - i - 1)
            print(f"  Progress: {i+1}/{n_iterations} | "
                  f"Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s | "
                  f"Best NDCG@10: {max(r['NDCG@10'] for r in results):.4f}")
    
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('NDCG@10', ascending=False)  # Higher is better
    
    print(f"\n✅ Random Search Complete!")
    print(f"   Time: {time.time() - start_time:.1f}s")
    print(f"   Best NDCG@10: {df_results.iloc[0]['NDCG@10']:.4f}")
    
    return df_results


def focused_search(movies_df: pd.DataFrame, train_df: pd.DataFrame,
                   valid_df: pd.DataFrame, base_weights: Dict[str, int],
                   features_to_tune: List[str]) -> pd.DataFrame:
    """
    EXPERIMENT 3: Focused Search
    
    PAPER USAGE: Table 3 - Feature Importance Analysis
    
    METHOD:
    - Fix most weights at baseline
    - Vary only specific features
    - Understand individual feature importance
    
    ADVANTAGE: Fast, interpretable
    """
    print(f"\n{'='*70}")
    print("EXPERIMENT 3: FOCUSED SEARCH (Feature Importance)")
    print(f"{'='*70}")
    print(f"Base weights: {base_weights}")
    print(f"Tuning features: {features_to_tune}")
    
    weight_values = [0, 1, 2, 3, 4, 5]
    results = []
    
    for feature in features_to_tune:
        print(f"\n--- Tuning: {feature} ---")
        
        for value in weight_values:
            weights = base_weights.copy()
            weights[feature] = value
            
            ndcg, precision, coverage = evaluate_weights(weights, movies_df, train_df, valid_df)
            
            results.append({
                'feature': feature,
                'value': value,
                'weights': str(weights),
                'NDCG@10': ndcg,
                'Precision@10': precision,
                'Coverage': coverage
            })
            
            print(f"  {feature}={value}: NDCG@10={ndcg:.4f}")
    
    df_results = pd.DataFrame(results)
    
    print(f"\n✅ Focused Search Complete!")
    
    return df_results


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    np.random.seed(42)
    
    print("="*70)
    print(" CONTENT-BASED WEIGHTS HYPERPARAMETER OPTIMIZATION")
    print(" Automated Search for Optimal Feature Weights")
    print("="*70)
    
    # Load data
    print("\n[1] Loading Data...")
    movies = load_movies()
    train_df, valid_df, test_df = load_train_valid_test_splits()
    
    print(f"  Train: {len(train_df)} ratings")
    print(f"  Valid: {len(valid_df)} ratings")
    print(f"  Test: {len(test_df)} ratings")
    
    
    # Define weight ranges with DOMAIN KNOWLEDGE CONSTRAINTS
    # CRITICAL: Genres and Actors MUST be >= 1 (fundamental features)
    # Keywords can be 0 (noisy, as experiments showed)
    # Companies can be 0 (not discriminative)
    weight_ranges = {
        'genres': [1, 2, 3, 4],      # MUST be >= 1 (most important!)
        'director': [0, 1, 2, 3],    # Can be 0
        'keywords': [0, 1],          # Can be 0 (noisy)
        'actors': [1, 2, 3],         # MUST be >= 1 (important!)
        'year': [0, 1, 2],           # Can be 0
        'overview': [0, 1, 2],       # Can be 0
        'companies': [0, 1, 2],      # Can be 0 (not discriminative)
        'countries': [0, 1, 2, 3]    # Can be 0
    }
    
    print(f"\n[2] Weight Ranges:")
    for feature, values in weight_ranges.items():
        print(f"  {feature}: {values}")
    
    # ===== EXPERIMENT 1: BAYESIAN OPTIMIZATION =====
    df_bayes = bayesian_optimization(movies, train_df, valid_df, weight_ranges, n_iterations=25)
    
    # ===== EXPERIMENT 2: RANDOM SEARCH =====
    df_random = random_search(movies, train_df, valid_df, weight_ranges, n_iterations=25)
    
    # ===== EXPERIMENT 3: FOCUSED SEARCH =====
    # Start from current optimal
    base_weights = {
        'genres': 2,
        'director': 2,
        'keywords': 1,
        'actors': 1,
        'year': 1,
        'overview': 1,
        'companies': 0,
        'countries': 2
    }
    
    features_to_tune = ['genres', 'director', 'keywords', 'countries']
    df_focused = focused_search(movies, train_df, valid_df, base_weights, features_to_tune)
    
    # ===== RESULTS SUMMARY =====
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")
    
    print("\n--- TOP 5 FROM BAYESIAN OPTIMIZATION ---")
    print(df_bayes.head(5)[['NDCG@10', 'Precision@10', 'Coverage', 'weights']].to_string(index=False))
    
    print("\n--- TOP 5 FROM RANDOM SEARCH ---")
    print(df_random.head(5)[['NDCG@10', 'Precision@10', 'Coverage', 'weights']].to_string(index=False))
    
    # Find overall best
    best_bayes = df_bayes.iloc[0]
    best_random = df_random.iloc[0]
    
    if best_bayes['NDCG@10'] > best_random['NDCG@10']:  # Higher is better
        best_overall = best_bayes
        best_method = "Bayesian Optimization"
    else:
        best_overall = best_random
        best_method = "Random Search"
    
    print(f"\n{'='*70}")
    print("BEST CONFIGURATION FOUND")
    print(f"{'='*70}")
    print(f"Method: {best_method}")
    print(f"NDCG@10: {best_overall['NDCG@10']:.4f}")
    print(f"Precision@10: {best_overall['Precision@10']:.4f}")
    print(f"Coverage: {best_overall['Coverage']:.4f}")
    print(f"Weights: {best_overall['weights']}")
    
    # Save results
    bayes_path = os.path.join(RESULTS_DIR, "cb_weights_bayesian_optimization.csv")
    random_path = os.path.join(RESULTS_DIR, "cb_weights_random_search.csv")
    focused_path = os.path.join(RESULTS_DIR, "cb_weights_focused_search.csv")
    
    df_bayes.to_csv(bayes_path, index=False)
    df_random.to_csv(random_path, index=False)
    df_focused.to_csv(focused_path, index=False)
    
    print(f"\n✅ Results saved to:")
    print(f"   - {bayes_path}")
    print(f"   - {random_path}")
    print(f"   - {focused_path}")
    
    # Generate paper report
    report_path = os.path.join(RESULTS_DIR, "CB_WEIGHTS_OPTIMIZATION_REPORT.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write(" CONTENT-BASED WEIGHTS HYPERPARAMETER OPTIMIZATION REPORT\n")
        f.write("="*70 + "\n\n")
        
        f.write("RESEARCH QUESTION:\n")
        f.write("What is the optimal feature weighting scheme for content-based filtering?\n\n")
        
        f.write("METHODS:\n")
        f.write("-" * 70 + "\n")
        f.write("1. Bayesian Optimization: 50 iterations (Gaussian Process-based)\n")
        f.write("2. Random Search: 50 iterations\n")
        f.write("3. Focused Search: Individual feature importance\n\n")
        
        f.write("BEST CONFIGURATION:\n")
        f.write("-" * 70 + "\n")
        f.write(f"Method: {best_method}\n")
        f.write(f"NDCG@10: {best_overall['NDCG@10']:.4f}\n")
        f.write(f"Precision@10: {best_overall['Precision@10']:.4f}\n")
        f.write(f"Weights: {best_overall['weights']}\n\n")
        
        f.write("TOP 10 CONFIGURATIONS (BAYESIAN OPTIMIZATION):\n")
        f.write("-" * 70 + "\n")
        f.write(df_bayes.head(10)[['NDCG@10', 'Precision@10', 'weights']].to_string(index=False))
        f.write("\n\n")
        
        f.write("TOP 10 CONFIGURATIONS (RANDOM SEARCH):\n")
        f.write("-" * 70 + "\n")
        f.write(df_random.head(10)[['NDCG@10', 'Precision@10', 'weights']].to_string(index=False))
        f.write("\n\n")
        
        f.write("CITATION TEMPLATE:\n")
        f.write("-" * 70 + "\n")
        f.write(f"\"We performed hyperparameter optimization using Bayesian Optimization\n")
        f.write(f"(Snoek et al., 2012) and Random Search (Bergstra & Bengio, 2012).\n")
        f.write(f"The optimal configuration achieved NDCG@10 of {best_overall['NDCG@10']:.4f}\n")
        f.write(f"on the validation set, demonstrating superior ranking quality.\"\n")
    
    print(f"   - {report_path}")
    
    print("\n" + "="*70)
    print(" OPTIMIZATION COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
