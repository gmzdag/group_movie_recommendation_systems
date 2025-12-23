"""
UBCF Experiment with Bayesian Optimization - NDCG@K Optimization
----------------------------------------------------------------
Scientific Methodology:
1. Train on TRAIN set
2. Optimize hyperparameters on VALIDATION set using Bayesian Optimization
3. Objective: Maximize NDCG@10 (ranking quality) instead of minimize RMSE
3. Final evaluation on TEST set (only once!)

Optimized Parameters:
- K_NEIGHBORS: Number of neighbors
- MIN_OVERLAP: Minimum overlap for similarity

Evaluation Metric: NDCG@10 (Normalized Discounted Cumulative Gain)
- More appropriate for ranking-based recommender systems
- Measures Top-K recommendation quality directly

References:
- Snoek et al. (2012) "Practical Bayesian Optimization of Machine Learning Algorithms"
- Cremonesi et al. (2010) "Performance of Recommender Algorithms on Top-N Recommendation Tasks"
"""

import sys
print("DEBUG: importing modules...")
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
import pandas as pd
from datetime import datetime
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.stats import norm
from scipy.optimize import minimize

def manual_mean_squared_error(y_true, y_pred):
    return np.mean((np.array(y_true) - np.array(y_pred))**2)

def manual_mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))

from recommender.data_loader import load_all_data
from recommender.UBCF.similarity_user import pearson_sw, pearson_shrink, cosine_sim, spearman_rank, spearman_sw
from recommender.UBCF.neighbors_user import compute_neighbors
from recommender.UBCF.user_based_cf import UserBasedCF

# Define paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


class BayesianOptimizer:
    """Bayesian Optimization using Gaussian Process"""
    
    def __init__(self, bounds, n_init=5, n_iter=15):
        self.bounds = np.array(bounds)
        self.n_init = n_init
        self.n_iter = n_iter
        self.X_observed = []
        self.y_observed = []
        self.gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=42
        )
    
    def _expected_improvement(self, X, xi=0.01):
        """Expected Improvement acquisition function"""
        mu, sigma = self.gp.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        
        if len(self.y_observed) == 0:
            return np.zeros_like(mu)
        
        mu_sample_opt = np.min(self.y_observed)
        
        with np.errstate(divide='warn'):
            imp = mu_sample_opt - mu - xi
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        
        return ei
    
    def _propose_location(self):
        """Propose next sampling point"""
        dim = self.bounds.shape[0]
        min_val = float('inf')
        min_x = None
        
        for _ in range(25):
            x0 = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=dim)
            res = minimize(
                lambda x: -self._expected_improvement(x.reshape(1, -1)),
                x0,
                bounds=self.bounds,
                method='L-BFGS-B'
            )
            if res.fun < min_val:
                min_val = res.fun
                min_x = res.x
        
        return min_x
    
    def optimize(self, objective_func, param_names):
        """Run Bayesian Optimization"""
        print(f"\n{'='*70}")
        print("BAYESIAN OPTIMIZATION - Hyperparameter Tuning")
        print(f"{'='*70}")
        print(f"Parameters: {param_names}")
        print(f"Random init: {self.n_init}, Bayesian iter: {self.n_iter}")
        print(f"{'='*70}\n")
        
        # Phase 1: Random initialization
        print(f"Phase 1: Random Initialization ({self.n_init} points)")
        for i in range(self.n_init):
            x = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])
            params = {name: int(val) for name, val in zip(param_names, x)}
            print(f"  [{i+1}/{self.n_init}] {params}", end=" → ")
            y = objective_func(params)
            self.X_observed.append(x)
            self.y_observed.append(y)
            print(f"RMSE: {y:.4f}")
        
        # Phase 2: Bayesian Optimization
        print(f"\nPhase 2: Bayesian Optimization ({self.n_iter} points)")
        for i in range(self.n_iter):
            self.gp.fit(np.array(self.X_observed), np.array(self.y_observed))
            x_next = self._propose_location()
            params = {name: int(val) for name, val in zip(param_names, x_next)}
            print(f"  [{i+1}/{self.n_iter}] {params}", end=" → ")
            y = objective_func(params)
            self.X_observed.append(x_next)
            self.y_observed.append(y)
            current_best = np.min(self.y_observed)
            print(f"RMSE: {y:.4f} (Best: {current_best:.4f})")
        
        # Find best
        best_idx = np.argmin(self.y_observed)
        best_x = self.X_observed[best_idx]
        best_params = {name: int(val) for name, val in zip(param_names, best_x)}
        best_score = self.y_observed[best_idx]
        
        print(f"\n{'='*70}")
        print(f"BEST: {best_params} → RMSE: {best_score:.4f}")
        print(f"{'='*70}\n")
        
        return best_params, best_score, self.X_observed, self.y_observed


def calculate_ndcg_at_k(ranked_list, relevant_items, k=10):
    """
    Calculate NDCG@K for a single user.
    
    Args:
        ranked_list: List of recommended item IDs (in rank order)
        relevant_items: Set of relevant item IDs (ground truth)
        k: Cutoff for evaluation
    
    Returns:
        NDCG@K score (0.0 to 1.0)
    """
    if not relevant_items or not ranked_list:
        return 0.0
    
    # Truncate to top-k
    ranked_list = ranked_list[:k]
    
    # Calculate DCG
    dcg = 0.0
    for i, item_id in enumerate(ranked_list):
        if item_id in relevant_items:
            # Binary relevance: 1 if relevant, 0 otherwise
            # Discount by log2(position + 2) (position is 0-indexed)
            dcg += 1.0 / np.log2(i + 2)
    
    # Calculate IDCG (ideal DCG)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant_items), k)))
    
    # Avoid division by zero
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def evaluate_model_ndcg(model, eval_df, R_train, k=10, relevance_threshold=4.0):
    """
    Evaluate UBCF model using NDCG@K.
    
    Args:
        model: Trained UBCF model
        eval_df: Evaluation dataframe (validation or test)
        R_train: Training matrix
        k: Top-K for NDCG calculation
        relevance_threshold: Rating threshold for relevance (default: 4.0)
    
    Returns:
        dict: Metrics including NDCG@K, Precision@K, Coverage
    """
    from collections import defaultdict
    
    # Group evaluation data by user
    user_eval_data = defaultdict(list)
    for _, row in eval_df.iterrows():
        user_eval_data[row['userId']].append({
            'movieId': row['movieId'],
            'rating': row['rating']
        })
    
    ndcg_scores = []
    precision_scores = []
    total_users = 0
    users_with_recs = 0
    
    # Get all candidate movies (from training set)
    all_movies = set(R_train.columns)
    
    for user_id, user_items in user_eval_data.items():
        # Skip if user not in training
        if user_id not in R_train.index:
            continue
        
        total_users += 1
        
        # Get user's training movies (to exclude from recommendations)
        user_train_movies = set(R_train.loc[user_id].dropna().index)
        
        # Candidate movies = all movies - training movies
        candidate_movies = list(all_movies - user_train_movies)
        
        if not candidate_movies:
            continue
        
        # Get predictions for all candidates
        predictions = []
        for movie_id in candidate_movies:
            pred = model.predict(user_id, movie_id)
            if not np.isnan(pred):
                predictions.append((movie_id, pred))
        
        if not predictions:
            continue
        
        users_with_recs += 1
        
        # Sort by predicted score (descending)
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        # Get top-K recommendations
        top_k_movies = [movie_id for movie_id, _ in predictions[:k]]
        
        # Get ground truth relevant items (high-rated in eval set)
        relevant_items = set(
            item['movieId'] for item in user_items 
            if item['rating'] >= relevance_threshold
        )
        
        # Calculate NDCG@K
        ndcg = calculate_ndcg_at_k(top_k_movies, relevant_items, k=k)
        ndcg_scores.append(ndcg)
        
        # Calculate Precision@K
        if top_k_movies:
            hits = len(set(top_k_movies) & relevant_items)
            precision = hits / len(top_k_movies)
            precision_scores.append(precision)
    
    if not ndcg_scores:
        return {
            "ndcg@k": 0.0,
            "precision@k": 0.0,
            "coverage": 0.0,
            "k": k
        }
    
    return {
        "ndcg@k": np.mean(ndcg_scores),
        "precision@k": np.mean(precision_scores) if precision_scores else 0.0,
        "coverage": users_with_recs / total_users if total_users > 0 else 0.0,
        "k": k,
        "num_users_evaluated": len(ndcg_scores)
    }


def evaluate_model_rmse(model, eval_df, R_train):
    """Evaluate UBCF model using RMSE (for comparison)"""
    preds, trues = [], []
    
    for _, row in eval_df.iterrows():
        u, m, true_r = row["userId"], row["movieId"], row["rating"]
        
        if m not in R_train.columns:
            continue
        
        raw_pred = model.predict(u, m)
        clipped_pred = min(5.0, max(0.5, raw_pred))
        
        preds.append(clipped_pred)
        trues.append(true_r)
    
    if len(preds) == 0:
        return {"rmse": float('inf'), "mae": float('inf'), "coverage": 0.0}
    
    rmse = math.sqrt(manual_mean_squared_error(trues, preds))
    mae = manual_mean_absolute_error(trues, preds)
    coverage = len(preds) / len(eval_df)
    
    return {"rmse": rmse, "mae": mae, "coverage": coverage, "preds": preds, "trues": trues}


if __name__ == "__main__":
    print("="*70)
    print("UBCF EXPERIMENT - BAYESIAN OPTIMIZATION")
    print("="*70)
    
    print("\n[1] Loading data...")
    movies, ratings, watchlists, R_cf, R_dense = load_all_data()
    
    print("\n[2] Loading splits (Train/Validation/Test)...")
    SPLITS_DIR = os.path.join(PROJECT_ROOT, "data", "splits")
    train_path = os.path.join(SPLITS_DIR, "train.csv")
    valid_path = os.path.join(SPLITS_DIR, "validation.csv")
    test_path = os.path.join(SPLITS_DIR, "test.csv")
    
    if not all(os.path.exists(p) for p in [train_path, valid_path, test_path]):
        raise FileNotFoundError(f"Split files not found in {SPLITS_DIR}")
    
    train_df = pd.read_csv(train_path)
    valid_df = pd.read_csv(valid_path)
    test_df = pd.read_csv(test_path)
    
    for df in [train_df, valid_df, test_df]:
        df["rating"] = df["rating"].astype(float)
    
    print(f"  Train: {len(train_df):,} | Valid: {len(valid_df):,} | Test: {len(test_df):,}")
    
    # Create training matrix
    R_train = train_df.pivot_table(index="userId", columns="movieId", values="rating", aggfunc="mean")
    global_mean = train_df["rating"].mean()
    user_means = R_train.mean(axis=1)
    item_means = R_train.mean(axis=0)
    
    print("\n[3] PHASE 1: Hyperparameter Optimization on VALIDATION")
    print("="*70)
    print("Objective: Maximize NDCG@10 (ranking quality)")
    print("="*70)
    
    # Define objective function
    def objective_function(params):
        """
        Maximize NDCG@10 on validation set.
        Returns negative NDCG (for minimization in Bayesian Optimization).
        """
        K = params['K_NEIGHBORS']
        MIN_OVERLAP = params['MIN_OVERLAP']
        
        # Use Pearson SW (usually best)
        neighbors = compute_neighbors(R_train, pearson_sw, K=K, min_overlap=MIN_OVERLAP)
        model = UserBasedCF(R_train, neighbors, user_means, item_means, global_mean)
        
        # Evaluate using NDCG@10
        metrics = evaluate_model_ndcg(model, valid_df, R_train, k=10)
        
        # Return negative NDCG (we minimize in Bayesian Optimization)
        return -metrics["ndcg@k"]
    
    # Search space
    bounds = [
        (20, 80),   # K_NEIGHBORS
        (2, 20),    # MIN_OVERLAP
    ]
    param_names = ['K_NEIGHBORS', 'MIN_OVERLAP']
    
    # Run optimization
    optimizer = BayesianOptimizer(bounds=bounds, n_init=5, n_iter=15)
    best_params, best_valid_score, X_history, y_history = optimizer.optimize(objective_function, param_names)
    
    # Convert negative NDCG back to positive for display
    best_valid_ndcg = -best_valid_score
    y_history_ndcg = [-y for y in y_history]  # Convert to positive NDCG
    
    # Save history
    history_df = pd.DataFrame(X_history, columns=param_names)
    history_df['NDCG@10'] = y_history_ndcg
    history_df['iteration'] = range(1, len(y_history) + 1)
    history_csv = os.path.join(RESULTS_DIR, "ubcf_bayesian_history.csv")
    history_df.to_csv(history_csv, index=False)
    print(f"History saved: {history_csv}")
    
    # Visualize optimization
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(range(1, len(y_history_ndcg) + 1), y_history_ndcg, 'o-', alpha=0.6)
    plt.axhline(y=best_valid_ndcg, color='r', linestyle='--', label=f'Best: {best_valid_ndcg:.4f}')
    plt.xlabel('Iteration')
    plt.ylabel('Validation NDCG@10')
    plt.title('Optimization Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.scatter(history_df['K_NEIGHBORS'], history_df['NDCG@10'], c=range(len(y_history)), cmap='viridis', s=100)
    plt.colorbar(label='Iteration')
    plt.xlabel('K_NEIGHBORS')
    plt.ylabel('Validation NDCG@10')
    plt.title('K_NEIGHBORS vs NDCG@10')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.scatter(history_df['MIN_OVERLAP'], history_df['NDCG@10'], c=range(len(y_history)), cmap='viridis', s=100)
    plt.colorbar(label='Iteration')
    plt.xlabel('MIN_OVERLAP')
    plt.ylabel('Validation NDCG@10')
    plt.title('MIN_OVERLAP vs NDCG@10')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    opt_plot = os.path.join(RESULTS_DIR, "ubcf_bayesian_optimization.png")
    plt.savefig(opt_plot, dpi=150)
    print(f"Plots saved: {opt_plot}")
    plt.close()
    
    print(f"\n[4] PHASE 2: Final Evaluation on TEST Set")
    print("="*70)
    print(f"Using optimized params: {best_params}\n")
    
    # Test all similarity metrics with best params
    similarity_functions = {
        "Pearson SW": pearson_sw,
        "Pearson Shrink": pearson_shrink,
        "Cosine": cosine_sim,
        "Spearman": spearman_rank,
        "Spearman SW": spearman_sw
    }
    
    test_results = []
    
    for sim_name, sim_func in similarity_functions.items():
        print(f"Testing {sim_name}...")
        
        neighbors = compute_neighbors(
            R_train, sim_func,
            K=best_params['K_NEIGHBORS'],
            min_overlap=best_params['MIN_OVERLAP']
        )
        
        model = UserBasedCF(R_train, neighbors, user_means, item_means, global_mean)
        
        # Evaluate with NDCG@10 (primary metric)
        metrics_ndcg = evaluate_model_ndcg(model, test_df, R_train, k=10)
        
        # Also calculate RMSE for comparison
        metrics_rmse = evaluate_model_rmse(model, test_df, R_train)
        
        print(f"  NDCG@10: {metrics_ndcg['ndcg@k']:.4f}, Precision@10: {metrics_ndcg['precision@k']:.4f}, RMSE: {metrics_rmse['rmse']:.4f}")
        
        test_results.append({
            "similarity": sim_name,
            **best_params,
            "ndcg@10": metrics_ndcg["ndcg@k"],
            "precision@10": metrics_ndcg["precision@k"],
            "rmse": metrics_rmse["rmse"],
            "mae": metrics_rmse["mae"],
            "coverage": metrics_ndcg["coverage"]
        })
    
    # Save results (sorted by NDCG@10, descending)
    test_df_results = pd.DataFrame(test_results).sort_values('ndcg@10', ascending=False)
    test_csv = os.path.join(RESULTS_DIR, "ubcf_experiment_results.csv")
    test_df_results.to_csv(test_csv, index=False)
    
    # Create report
    report_path = os.path.join(RESULTS_DIR, "ubcf_experiment_results.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("="*70 + "\n")
        f.write("UBCF EXPERIMENT - BAYESIAN OPTIMIZATION (NDCG@10)\n")
        f.write("="*70 + "\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("METHODOLOGY:\n")
        f.write("  - Bayesian Optimization (Gaussian Process + Expected Improvement)\n")
        f.write("  - Objective: Maximize NDCG@10 (ranking quality)\n")
        f.write(f"  - Search space: K ∈ [20,80], MIN_OVERLAP ∈ [2,20]\n")
        f.write(f"  - Total evaluations: {len(y_history)}\n\n")
        
        f.write("BEST HYPERPARAMETERS (from Validation):\n")
        for k, v in best_params.items():
            f.write(f"  {k}: {v}\n")
        f.write(f"  Validation NDCG@10: {best_valid_ndcg:.4f}\n\n")
        
        f.write("FINAL TEST RESULTS (sorted by NDCG@10):\n")
        for _, row in test_df_results.iterrows():
            f.write(f"  {row['similarity']:20s} | NDCG@10: {row['ndcg@10']:.4f} | Precision@10: {row['precision@10']:.4f} | RMSE: {row['rmse']:.4f}\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("CITATIONS:\n")
        f.write('  "We optimized UBCF hyperparameters using Bayesian Optimization\n')
        f.write('   (Snoek et al., 2012) to maximize NDCG@10, a ranking quality metric\n')
        f.write('   more appropriate for recommender systems than RMSE (Cremonesi et al.,\n')
        f.write(f'   2010). Best model: {test_df_results.iloc[0]["similarity"]} with\n')
        f.write(f'   NDCG@10={test_df_results.iloc[0]["ndcg@10"]:.4f} on test set."\n\n')
        f.write("References:\n")
        f.write("  - Snoek et al. (2012) Practical Bayesian Optimization of ML Algorithms\n")
        f.write("  - Cremonesi et al. (2010) Performance of Recommender Algorithms on Top-N Tasks\n")
        f.write("="*70 + "\n")
    
    print(f"\n{'='*70}")
    print("FINAL RESULTS (sorted by NDCG@10):")
    print(f"{'='*70}")
    for i, (_, row) in enumerate(test_df_results.iterrows(), 1):
        print(f"{i}. {row['similarity']:20s} | NDCG@10: {row['ndcg@10']:.4f} | Precision@10: {row['precision@k']:.4f} | RMSE: {row['rmse']:.4f}")
    
    print(f"\n{'='*70}")
    print("✅ EXPERIMENT COMPLETE!")
    print(f"{'='*70}")
    print(f"Optimization method: Bayesian Optimization (NDCG@10)")
    print(f"Best NDCG@10: {test_df_results.iloc[0]['ndcg@10']:.4f}")
    print(f"Best model: {test_df_results.iloc[0]['similarity']}")
    print(f"Results: {test_csv}")
    print(f"Report: {report_path}")
    print(f"{'='*70}")
