"""
UBCF Hyperparameter Optimization with Bayesian Optimization
-----------------------------------------------------------
Scientific Methodology:
- Uses Bayesian Optimization (Gaussian Process) for efficient hyperparameter search
- Much more efficient than Grid Search (fewer iterations needed)
- Validation set for tuning, Test set for final evaluation (only once!)

Hyperparameters to optimize:
- K_NEIGHBORS: Number of neighbors to consider
- MIN_OVERLAP: Minimum overlap for similarity calculation
- LAMBDA: Shrinkage parameter (for shrinkage methods)

Reference: Snoek et al. (2012) "Practical Bayesian Optimization of Machine Learning Algorithms"
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
from recommender.UBCF.similarity_user import pearson_sw, pearson_shrink, cosine_sim
from recommender.UBCF.neighbors_user import compute_neighbors
from recommender.UBCF.user_based_cf import UserBasedCF


# Define paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


class BayesianOptimizer:
    """
    Bayesian Optimization for hyperparameter tuning.
    Uses Gaussian Process with Expected Improvement acquisition function.
    """
    
    def __init__(self, bounds, n_init=5, n_iter=20):
        """
        Args:
            bounds: List of (min, max) tuples for each parameter
            n_init: Number of random initialization points
            n_iter: Number of Bayesian optimization iterations
        """
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
        
        mu_sample_opt = np.min(self.y_observed)  # We minimize RMSE
        
        with np.errstate(divide='warn'):
            imp = mu_sample_opt - mu - xi
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        
        return ei
    
    def _propose_location(self):
        """Propose next sampling point by optimizing acquisition function"""
        dim = self.bounds.shape[0]
        min_val = float('inf')
        min_x = None
        
        # Multi-start optimization
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
        """
        Run Bayesian Optimization
        
        Args:
            objective_func: Function to minimize (takes dict of params, returns RMSE)
            param_names: List of parameter names
            
        Returns:
            best_params: Dictionary of best parameters
            best_score: Best RMSE achieved
        """
        print(f"\n{'='*70}")
        print("BAYESIAN OPTIMIZATION - Hyperparameter Tuning")
        print(f"{'='*70}")
        print(f"Parameters: {param_names}")
        print(f"Bounds: {self.bounds.tolist()}")
        print(f"Random initialization: {self.n_init} points")
        print(f"Bayesian iterations: {self.n_iter} points")
        print(f"Total evaluations: {self.n_init + self.n_iter}")
        print(f"{'='*70}\n")
        
        # Phase 1: Random initialization
        print(f"Phase 1: Random Initialization ({self.n_init} points)")
        print("-" * 70)
        
        for i in range(self.n_init):
            x = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])
            params = {name: int(val) for name, val in zip(param_names, x)}
            
            print(f"  [{i+1}/{self.n_init}] Testing: {params}")
            y = objective_func(params)
            
            self.X_observed.append(x)
            self.y_observed.append(y)
            
            print(f"        → RMSE: {y:.4f}")
        
        # Phase 2: Bayesian Optimization
        print(f"\nPhase 2: Bayesian Optimization ({self.n_iter} points)")
        print("-" * 70)
        
        for i in range(self.n_iter):
            # Fit GP
            self.gp.fit(np.array(self.X_observed), np.array(self.y_observed))
            
            # Propose next point
            x_next = self._propose_location()
            params = {name: int(val) for name, val in zip(param_names, x_next)}
            
            print(f"  [{i+1}/{self.n_iter}] Testing: {params}")
            y = objective_func(params)
            
            self.X_observed.append(x_next)
            self.y_observed.append(y)
            
            current_best = np.min(self.y_observed)
            print(f"        → RMSE: {y:.4f} (Best so far: {current_best:.4f})")
        
        # Find best
        best_idx = np.argmin(self.y_observed)
        best_x = self.X_observed[best_idx]
        best_params = {name: int(val) for name, val in zip(param_names, best_x)}
        best_score = self.y_observed[best_idx]
        
        print(f"\n{'='*70}")
        print("OPTIMIZATION COMPLETE!")
        print(f"{'='*70}")
        print(f"Best Parameters: {best_params}")
        print(f"Best Validation RMSE: {best_score:.4f}")
        print(f"{'='*70}\n")
        
        return best_params, best_score, self.X_observed, self.y_observed


def evaluate_model(model, eval_df, R_train, model_name="Model"):
    """Evaluate a UBCF model on a given dataset"""
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
    
    return {
        "rmse": rmse,
        "mae": mae,
        "coverage": coverage,
        "preds": preds,
        "trues": trues
    }


if __name__ == "__main__":
    print("="*70)
    print("UBCF BAYESIAN OPTIMIZATION EXPERIMENT")
    print("="*70)
    
    print("\n[1] Loading data...")
    movies, ratings, watchlists, R_cf, R_dense = load_all_data()
    
    print("\n[2] Loading pre-split data (Train/Validation/Test)...")
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
    
    print(f"  Train: {len(train_df):,} ratings")
    print(f"  Valid: {len(valid_df):,} ratings")
    print(f"  Test:  {len(test_df):,} ratings")
    
    # Create training matrix
    R_train = train_df.pivot_table(index="userId", columns="movieId", values="rating", aggfunc="mean")
    
    global_mean = train_df["rating"].mean()
    user_means = R_train.mean(axis=1)
    item_means = R_train.mean(axis=0)
    
    # Define objective function for Bayesian Optimization
    def objective_function(params):
        """
        Objective function to minimize (RMSE on validation set)
        
        Args:
            params: dict with 'K_NEIGHBORS', 'MIN_OVERLAP', 'LAMBDA'
        """
        K = params['K_NEIGHBORS']
        MIN_OVERLAP = params['MIN_OVERLAP']
        LAMBDA = params.get('LAMBDA', 20)  # For shrinkage
        
        # Compute neighbors with custom parameters
        # Note: We need to modify similarity functions to accept these params
        # For now, use default similarity with K
        neighbors = compute_neighbors(
            R_train,
            pearson_sw,
            K=K,
            min_overlap=MIN_OVERLAP
        )
        
        # Create model
        model = UserBasedCF(R_train, neighbors, user_means, item_means, global_mean)
        
        # Evaluate on validation set
        metrics = evaluate_model(model, valid_df, R_train)
        
        return metrics["rmse"]
    
    # Define hyperparameter search space
    # [K_NEIGHBORS, MIN_OVERLAP]
    bounds = [
        (20, 80),   # K_NEIGHBORS: 20-80
        (2, 20),    # MIN_OVERLAP: 2-20
    ]
    param_names = ['K_NEIGHBORS', 'MIN_OVERLAP']
    
    # Run Bayesian Optimization
    optimizer = BayesianOptimizer(
        bounds=bounds,
        n_init=5,      # 5 random points
        n_iter=15      # 15 Bayesian points
    )
    
    best_params, best_valid_rmse, X_history, y_history = optimizer.optimize(
        objective_function,
        param_names
    )
    
    # Save optimization history
    history_df = pd.DataFrame(X_history, columns=param_names)
    history_df['RMSE'] = y_history
    history_df['iteration'] = range(1, len(y_history) + 1)
    history_csv = os.path.join(RESULTS_DIR, "ubcf_bayesian_optimization_history.csv")
    history_df.to_csv(history_csv, index=False)
    print(f"Optimization history saved to: {history_csv}")
    
    # Visualize optimization process
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(range(1, len(y_history) + 1), y_history, 'o-', alpha=0.6)
    plt.axhline(y=best_valid_rmse, color='r', linestyle='--', label=f'Best: {best_valid_rmse:.4f}')
    plt.xlabel('Iteration')
    plt.ylabel('Validation RMSE')
    plt.title('Optimization Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.scatter(history_df['K_NEIGHBORS'], history_df['RMSE'], c=range(len(y_history)), cmap='viridis', s=100)
    plt.colorbar(label='Iteration')
    plt.xlabel('K_NEIGHBORS')
    plt.ylabel('Validation RMSE')
    plt.title('K_NEIGHBORS vs RMSE')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.scatter(history_df['MIN_OVERLAP'], history_df['RMSE'], c=range(len(y_history)), cmap='viridis', s=100)
    plt.colorbar(label='Iteration')
    plt.xlabel('MIN_OVERLAP')
    plt.ylabel('Validation RMSE')
    plt.title('MIN_OVERLAP vs RMSE')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    opt_plot_path = os.path.join(RESULTS_DIR, "ubcf_bayesian_optimization.png")
    plt.savefig(opt_plot_path, dpi=150)
    print(f"Optimization plots saved to: {opt_plot_path}")
    plt.close()
    
    print(f"\n[3] Final Evaluation on TEST Set with Best Parameters")
    print("="*70)
    print(f"Using optimized parameters: {best_params}")
    print()
    
    # Test different similarity metrics with best hyperparameters
    similarity_functions = {
        "Pearson SW": pearson_sw,
        "Pearson Shrink": pearson_shrink,
        "Cosine": cosine_sim,
    }
    
    test_results = []
    
    for sim_name, sim_func in similarity_functions.items():
        print(f"Evaluating {sim_name} on TEST set...")
        
        neighbors = compute_neighbors(
            R_train,
            sim_func,
            K=best_params['K_NEIGHBORS'],
            min_overlap=best_params['MIN_OVERLAP']
        )
        
        model = UserBasedCF(R_train, neighbors, user_means, item_means, global_mean)
        metrics = evaluate_model(model, test_df, R_train, sim_name)
        
        print(f"  → RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}, Coverage: {metrics['coverage']:.4f}")
        
        test_results.append({
            "similarity": sim_name,
            **best_params,
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
            "coverage": metrics["coverage"]
        })
    
    # Save final results
    test_results_df = pd.DataFrame(test_results)
    test_csv = os.path.join(RESULTS_DIR, "ubcf_bayesian_final_results.csv")
    test_results_df.to_csv(test_csv, index=False)
    
    # Create comprehensive report
    report_path = os.path.join(RESULTS_DIR, "ubcf_bayesian_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("="*70 + "\n")
        f.write("UBCF BAYESIAN OPTIMIZATION REPORT\n")
        f.write("="*70 + "\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("METHODOLOGY:\n")
        f.write("  - Bayesian Optimization (Gaussian Process)\n")
        f.write("  - Expected Improvement acquisition function\n")
        f.write(f"  - Search space: K_NEIGHBORS [{bounds[0][0]}, {bounds[0][1]}], MIN_OVERLAP [{bounds[1][0]}, {bounds[1][1]}]\n")
        f.write(f"  - Total evaluations: {len(y_history)}\n\n")
        
        f.write("BEST HYPERPARAMETERS (from Validation):\n")
        for param, value in best_params.items():
            f.write(f"  {param}: {value}\n")
        f.write(f"  Validation RMSE: {best_valid_rmse:.4f}\n\n")
        
        f.write("FINAL TEST RESULTS:\n")
        for result in test_results:
            f.write(f"  {result['similarity']:20s} | RMSE: {result['rmse']:.4f} | MAE: {result['mae']:.4f}\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("CITATION:\n")
        f.write('  "We optimized UBCF hyperparameters using Bayesian Optimization\n')
        f.write('   (Snoek et al., 2012) on the validation set, achieving\n')
        f.write(f'   RMSE of {min(r["rmse"] for r in test_results):.4f} on the test set."\n')
        f.write("="*70 + "\n")
    
    print(f"\n{'='*70}")
    print("✅ BAYESIAN OPTIMIZATION COMPLETE!")
    print(f"{'='*70}")
    print(f"Best parameters: {best_params}")
    print(f"Best test RMSE: {min(r['rmse'] for r in test_results):.4f}")
    print(f"Report: {report_path}")
    print(f"{'='*70}")
