
import sys
import os
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add parent directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from recommender.data_loader import load_train_valid_test_splits
from recommender.UBCF.similarity_user import pearson_sw, pearson_shrink, cosine_sim
from recommender.UBCF.neighbors_user import load_or_compute_neighbors
from recommender.UBCF.user_based_cf import UserBasedCF


# Define paths relative to the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# Use specific cache files for fixed splits
CACHE_SW = os.path.join(CACHE_DIR, "user_neighbors_sw_fixed.pkl")
CACHE_SHR = os.path.join(CACHE_DIR, "user_neighbors_shrink_fixed.pkl")
CACHE_COS = os.path.join(CACHE_DIR, "user_neighbors_cosine_fixed.pkl")

def manual_mean_squared_error(y_true, y_pred):
    return np.mean((np.array(y_true) - np.array(y_pred))**2)

def manual_mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))

def evaluate_model(model, test_df, R_train_columns):
    preds, trues = [], []
    
    # We only predict for items that exist in training set (or handle new items via generic means, here we skip)
    # New users will get global mean if they are not in R_train (handled by UserBasedCF logic)
    
    # Pre-check columns for speed
    train_items = set(R_train_columns)
    
    # Iterate
    # Note: iterating rows is slow, but acceptable for this scale
    for _, row in test_df.iterrows():
        u, m, true_r = row["userId"], row["movieId"], row["rating"]
        
        # If movie is completely unknown to the model (not in R_train), we can't do CF.
        # Options: Skip or predict global mean.
        # UserBasedCF returns global_mean if movie not in R.columns.
        if m not in train_items:
             # If we want to strictly evaluate CF power on known items:
             # continue
             # But for a system eval, we should probably include it (it will be an error due to cold start)
             pass 

        raw_pred = model.predict(u, m) 
        
        # Clip
        clipped_pred = min(5.0, max(0.5, raw_pred))
        
        preds.append(clipped_pred)
        trues.append(true_r)
        
    rmse = math.sqrt(manual_mean_squared_error(trues, preds))
    mae = manual_mean_absolute_error(trues, preds)
    coverage = len(preds) / len(test_df)
    
    return rmse, mae, coverage, preds, trues

if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    print("[1] Loading fixed splits...")
    try:
        train_df, valid_df, test_df = load_train_valid_test_splits()
    except FileNotFoundError as e:
        print(e)
        print("Please run src/experiments/create_temporal_splits.py first.")
        sys.exit(1)
        
    print(f"Train: {len(train_df)}, Valid: {len(valid_df)}, Test: {len(test_df)}")

    # Build R_train
    # Pivot table. Note: 'rating' is column name.
    print("[2] Building Training Matrix...")
    R_train = train_df.pivot_table(index="userId", columns="movieId", values="rating", aggfunc="mean")
    
    # Calculate means
    global_mean = train_df["rating"].mean()
    user_means = R_train.mean(axis=1)
    item_means = R_train.mean(axis=0)

    print("[3] Computing/Loading Neighbors (Fixed Split)...")
    neighbors_sw = load_or_compute_neighbors(CACHE_SW, R_train, pearson_sw)
    neighbors_shr = load_or_compute_neighbors(CACHE_SHR, R_train, pearson_shrink)
    neighbors_cos = load_or_compute_neighbors(CACHE_COS, R_train, cosine_sim)

    print("[4] Initializing Models...")
    model_sw  = UserBasedCF(R_train, neighbors_sw,  user_means, item_means, global_mean)
    model_shr = UserBasedCF(R_train, neighbors_shr, user_means, item_means, global_mean)
    model_cos = UserBasedCF(R_train, neighbors_cos, user_means, item_means, global_mean)

    print("[5] Evaluating on TEST set...")
    # Evaluate SW
    print("Evaluating Significance Weighting...")
    rmse_sw, mae_sw, cov_sw, preds_sw, trues_sw = evaluate_model(model_sw, test_df, R_train.columns)
    
    # Evaluate Shrinkage
    print("Evaluating Shrinkage...")
    rmse_shr, mae_shr, cov_shr, preds_shr, trues_shr = evaluate_model(model_shr, test_df, R_train.columns)
    
    # Evaluate Cosine
    print("Evaluating Cosine...")
    rmse_cos, mae_cos, cov_cos, preds_cos, trues_cos = evaluate_model(model_cos, test_df, R_train.columns)

    # Printing Results
    output = []
    output.append("=== FIXED TEMPORAL SPLIT EVALUATION ===")
    output.append(f"Train samples: {len(train_df)}")
    output.append(f"Test samples : {len(test_df)}")
    output.append("-" * 30)
    output.append("Method               RMSE    MAE     Coverage")
    output.append("-" * 30)
    output.append(f"Pearson SW           {rmse_sw:.4f}  {mae_sw:.4f}  {cov_sw:.2%}")
    output.append(f"Pearson Shrinkage    {rmse_shr:.4f}  {mae_shr:.4f}  {cov_shr:.2%}")
    output.append(f"Cosine Similarity    {rmse_cos:.4f}  {mae_cos:.4f}  {cov_cos:.2%}")
    output.append("-" * 30)
    
    print("\n".join(output))
    
    # Save results
    res_path = os.path.join(RESULTS_DIR, "fixed_split_results.txt")
    with open(res_path, "w") as f:
        f.write("\n".join(output))
        
    print(f"Results saved to {res_path}")
    
    # Plotting
    plt.figure(figsize=(18,5))
    
    plt.subplot(1,3,1)
    plt.hist(np.array(preds_sw) - np.array(trues_sw), bins=30, alpha=0.7)
    plt.title(f"Pearson SW (RMSE={rmse_sw:.3f})")
    
    plt.subplot(1,3,2)
    plt.hist(np.array(preds_shr) - np.array(trues_shr), bins=30, alpha=0.7, color='orange')
    plt.title(f"Shrinkage (RMSE={rmse_shr:.3f})")

    plt.subplot(1,3,3)
    plt.hist(np.array(preds_cos) - np.array(trues_cos), bins=30, alpha=0.7, color='green')
    plt.title(f"Cosine (RMSE={rmse_cos:.3f})")
    
    plot_path = os.path.join(RESULTS_DIR, "fixed_split_error_dist.png")
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
    plt.close()
