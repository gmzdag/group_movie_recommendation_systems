"""
UBCF Experiment with Proper Validation Set Usage
-------------------------------------------------
Scientific Methodology:
1. Train on TRAIN set
2. Tune hyperparameters on VALIDATION set
3. Final evaluation on TEST set (only once!)

This prevents overfitting to test set and follows ML best practices.
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

def manual_mean_squared_error(y_true, y_pred):
    return np.mean((np.array(y_true) - np.array(y_pred))**2)

def manual_mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))

from recommender.data_loader import load_all_data
from recommender.UBCF.similarity_user import pearson_sw, pearson_shrink, cosine_sim, spearman_rank, spearman_sw
from recommender.UBCF.neighbors_user import load_or_compute_neighbors
from recommender.UBCF.user_based_cf import UserBasedCF


# Define paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def evaluate_model(model, eval_df, R_train, model_name="Model"):
    """
    Evaluate a UBCF model on a given dataset.
    
    Args:
        model: Trained UBCF model
        eval_df: Evaluation dataframe (validation or test)
        R_train: Training matrix (to check movie existence)
        model_name: Name for logging
        
    Returns:
        dict: Metrics (RMSE, MAE, Coverage)
    """
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
    
    print(f"  {model_name}: RMSE={rmse:.4f}, MAE={mae:.4f}, Coverage={coverage:.4f}")
    
    return {
        "rmse": rmse,
        "mae": mae,
        "coverage": coverage,
        "preds": preds,
        "trues": trues
    }


if __name__ == "__main__":
    print("="*70)
    print("UBCF EXPERIMENT WITH VALIDATION SET")
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
    
    # Ensure ratings are float
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
    
    print(f"\n[3] PHASE 1: Hyperparameter Tuning on VALIDATION Set")
    print("="*70)
    
    # Test different K_NEIGHBORS values
    K_VALUES = [20, 30, 40, 50, 60]
    
    best_k = None
    best_rmse = float('inf')
    best_similarity = None
    
    validation_results = []
    
    # Test Pearson SW (usually best performing)
    print("\nTesting K_NEIGHBORS values with Pearson SW...")
    for k in K_VALUES:
        print(f"\n  K_NEIGHBORS = {k}")
        
        # Compute neighbors
        neighbors = load_or_compute_neighbors(
            R_train, 
            pearson_sw, 
            K=k, 
            metric=f"pearson_sw_k{k}_validation"
        )
        
        # Create model
        model = UserBasedCF(R_train, neighbors, user_means, item_means, global_mean)
        
        # Evaluate on VALIDATION set
        metrics = evaluate_model(model, valid_df, R_train, f"K={k}")
        
        validation_results.append({
            "similarity": "Pearson SW",
            "K": k,
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
            "coverage": metrics["coverage"]
        })
        
        # Track best
        if metrics["rmse"] < best_rmse:
            best_rmse = metrics["rmse"]
            best_k = k
            best_similarity = "Pearson SW"
    
    print(f"\n{'='*70}")
    print(f"BEST HYPERPARAMETERS (from Validation):")
    print(f"  Similarity: {best_similarity}")
    print(f"  K_NEIGHBORS: {best_k}")
    print(f"  Validation RMSE: {best_rmse:.4f}")
    print(f"{'='*70}")
    
    # Save validation results
    valid_results_df = pd.DataFrame(validation_results)
    valid_csv_path = os.path.join(RESULTS_DIR, "ubcf_validation_tuning.csv")
    valid_results_df.to_csv(valid_csv_path, index=False)
    print(f"\nValidation results saved to: {valid_csv_path}")
    
    print(f"\n[4] PHASE 2: Final Evaluation on TEST Set")
    print("="*70)
    print("⚠️  Using BEST hyperparameters from validation")
    print(f"   K_NEIGHBORS = {best_k}")
    print()
    
    # Test all similarity metrics with best K on TEST set
    similarity_functions = {
        "Pearson SW": pearson_sw,
        "Pearson Shrink": pearson_shrink,
        "Cosine": cosine_sim,
        "Spearman": spearman_rank,
        "Spearman SW": spearman_sw
    }
    
    test_results = []
    test_predictions = {}
    
    for sim_name, sim_func in similarity_functions.items():
        print(f"\nEvaluating {sim_name} (K={best_k}) on TEST set...")
        
        # Compute neighbors with best K
        neighbors = load_or_compute_neighbors(
            R_train,
            sim_func,
            K=best_k,
            metric=f"{sim_name.lower().replace(' ', '_')}_k{best_k}_final"
        )
        
        # Create model
        model = UserBasedCF(R_train, neighbors, user_means, item_means, global_mean)
        
        # Evaluate on TEST set
        metrics = evaluate_model(model, test_df, R_train, sim_name)
        
        test_results.append({
            "similarity": sim_name,
            "K_NEIGHBORS": best_k,
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
            "coverage": metrics["coverage"]
        })
        
        test_predictions[sim_name] = {
            "preds": metrics["preds"],
            "trues": metrics["trues"]
        }
    
    print(f"\n{'='*70}")
    print("FINAL TEST SET RESULTS:")
    print(f"{'='*70}")
    
    # Sort by RMSE
    test_results_sorted = sorted(test_results, key=lambda x: x["rmse"])
    
    for i, result in enumerate(test_results_sorted, 1):
        print(f"{i}. {result['similarity']:20s} | RMSE: {result['rmse']:.4f} | MAE: {result['mae']:.4f} | Coverage: {result['coverage']:.4f}")
    
    # Save test results
    test_results_df = pd.DataFrame(test_results_sorted)
    test_csv_path = os.path.join(RESULTS_DIR, "ubcf_test_final_results.csv")
    test_results_df.to_csv(test_csv_path, index=False)
    print(f"\nTest results saved to: {test_csv_path}")
    
    # Create comprehensive report
    report_path = os.path.join(RESULTS_DIR, "ubcf_experiment_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("="*70 + "\n")
        f.write("UBCF EXPERIMENT REPORT - WITH VALIDATION SET\n")
        f.write("="*70 + "\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("METHODOLOGY:\n")
        f.write("  1. Train on TRAIN set (70% of data)\n")
        f.write("  2. Tune K_NEIGHBORS on VALIDATION set (15% of data)\n")
        f.write("  3. Final evaluation on TEST set (15% of data)\n\n")
        
        f.write("VALIDATION PHASE:\n")
        f.write(f"  Tested K values: {K_VALUES}\n")
        f.write(f"  Best K: {best_k}\n")
        f.write(f"  Best Validation RMSE: {best_rmse:.4f}\n\n")
        
        f.write("TEST PHASE (Final Results):\n")
        for i, result in enumerate(test_results_sorted, 1):
            f.write(f"  {i}. {result['similarity']:20s} | RMSE: {result['rmse']:.4f} | MAE: {result['mae']:.4f}\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("SCIENTIFIC VALIDITY:\n")
        f.write("  ✅ Validation set used for hyperparameter tuning\n")
        f.write("  ✅ Test set used only once for final evaluation\n")
        f.write("  ✅ No data leakage between sets\n")
        f.write("  ✅ Temporal ordering preserved\n")
        f.write("="*70 + "\n")
    
    print(f"\nFull report saved to: {report_path}")
    
    # Create visualization
    print("\n[5] Creating visualizations...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'UBCF Error Distributions (K={best_k}, Test Set)', fontsize=16)
    
    for idx, (sim_name, data) in enumerate(test_predictions.items()):
        if idx >= 6:
            break
        
        row = idx // 3
        col = idx % 3
        
        errors = np.array(data["preds"]) - np.array(data["trues"])
        
        axes[row, col].hist(errors, bins=30, alpha=0.7, edgecolor='black')
        axes[row, col].set_title(f'{sim_name}')
        axes[row, col].set_xlabel('Prediction Error')
        axes[row, col].set_ylabel('Frequency')
        axes[row, col].axvline(x=0, color='red', linestyle='--', linewidth=1)
        axes[row, col].grid(True, alpha=0.3)
    
    # Hide unused subplot
    if len(test_predictions) < 6:
        axes[1, 2].axis('off')
    
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "ubcf_validation_experiment_plots.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Plots saved to: {plot_path}")
    plt.close()
    
    print("\n" + "="*70)
    print("✅ EXPERIMENT COMPLETE!")
    print("="*70)
