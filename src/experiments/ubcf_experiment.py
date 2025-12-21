"""
Evaluation script for UBCF with:
- significance weighting
- shrinkage pearson
"""

import sys
print("DEBUG: importing modules...")
import os
# Add parent directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
# from sklearn.metrics import mean_squared_error, mean_absolute_error
# from sklearn.model_selection import train_test_split
import pandas as pd

def manual_mean_squared_error(y_true, y_pred):
    return np.mean((np.array(y_true) - np.array(y_pred))**2)

def manual_mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))

def manual_train_test_split(df, test_size=0.2, random_state=42):
    np.random.seed(random_state)
    mask = np.random.rand(len(df)) < (1 - test_size)
    train = df[mask]
    test = df[~mask]
    return train, test

from recommender.data_loader import load_all_data
from recommender.UBCF.similarity_user import pearson_sw, pearson_shrink, cosine_sim, spearman_rank, spearman_sw
from recommender.UBCF.neighbors_user import load_or_compute_neighbors
from recommender.UBCF.user_based_cf import UserBasedCF


# Define paths relative to the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")

CACHE_SW = os.path.join(CACHE_DIR, "user_neighbors_sw.pkl")
CACHE_SHR = os.path.join(CACHE_DIR, "user_neighbors_shrink.pkl")
CACHE_COS = os.path.join(CACHE_DIR, "user_neighbors_cosine.pkl")



if __name__ == "__main__":
    print("[1] Loading data...")
    movies, ratings, watchlists, R_cf, R_dense = load_all_data()

    # SUBSET USERS FOR SPEED (Experiment)
    # ratings = ratings[ratings["userId"] <= 100]

    ratings_clean = ratings.groupby(["userId","movieId"])["rating"].mean().reset_index()

    print("[2] Loading pre-split data...")
    SPLITS_DIR = os.path.join(PROJECT_ROOT, "data", "splits")
    train_path = os.path.join(SPLITS_DIR, "train.csv")
    test_path = os.path.join(SPLITS_DIR, "test.csv")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"Split files not found in {SPLITS_DIR}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Ensure ratings are float
    train_df["rating"] = train_df["rating"].astype(float)
    test_df["rating"] = test_df["rating"].astype(float)

    R_train = train_df.pivot(index="userId", columns="movieId", values="rating")
    # Do NOT reindex to full dataset for this subsets experiment
    # R_train = R_train.reindex(index=R_cf.index, columns=R_cf.columns)

    global_mean = train_df["rating"].mean()
    user_means = R_train.mean(axis=1)
    item_means = R_train.mean(axis=0)

    print("[3] Loading neighbors with OPTIMIZED parameters (Defaults updated)...")
    # Defaults in similarity_user.py are now MIN_OVERLAP=10, K=20, LAMBDA=20
    
    # Use K=50 for neighbors
    K_NEIGHBORS = 50
    
    neighbors_sw = load_or_compute_neighbors(R_train, pearson_sw, K=K_NEIGHBORS, metric="pearson_sw_opt_def")
    neighbors_shr = load_or_compute_neighbors(R_train, pearson_shrink, K=K_NEIGHBORS, metric="pearson_shr_opt_def")
    neighbors_cos = load_or_compute_neighbors(R_train, cosine_sim, K=K_NEIGHBORS, metric="cosine_opt_def")
    neighbors_spearman = load_or_compute_neighbors(R_train, spearman_rank, K=K_NEIGHBORS, metric="spearman_opt_def")
    neighbors_spearman_sw = load_or_compute_neighbors(R_train, spearman_sw, K=K_NEIGHBORS, metric="spearman_sw_opt_def")

    model_sw  = UserBasedCF(R_train, neighbors_sw,  user_means, item_means, global_mean)
    model_shr = UserBasedCF(R_train, neighbors_shr, user_means, item_means, global_mean)
    model_cos = UserBasedCF(R_train, neighbors_cos, user_means, item_means, global_mean)
    model_spearman = UserBasedCF(R_train, neighbors_spearman, user_means, item_means, global_mean)
    model_spearman_sw = UserBasedCF(R_train, neighbors_spearman_sw, user_means, item_means, global_mean)


    # ----------- Evaluation -----------
    def evaluate(model):
        preds, trues = [], []
        for _, row in test_df.iterrows():
            u, m, true_r = row["userId"], row["movieId"], row["rating"]

            if m not in R_train.columns:
                continue

            raw_pred = model.predict(u, m)
            # Clip prediction to valid range for fair error calculation
            clipped_pred = min(5.0, max(0.5, raw_pred))
            
            preds.append(clipped_pred)
            trues.append(true_r)

        rmse = math.sqrt(manual_mean_squared_error(trues, preds))
        mae = manual_mean_absolute_error(trues, preds)
        coverage = len(preds) / len(test_df)

        return rmse, mae, coverage, preds, trues


    print("[4] Testing models...")

    rmse_sw, mae_sw, cov_sw, preds_sw, trues_sw = evaluate(model_sw)
    rmse_shr, mae_shr, cov_shr, preds_shr, trues_shr = evaluate(model_shr)
    rmse_cos, mae_cos, cov_cos, preds_cos, trues_cos = evaluate(model_cos)
    rmse_sp, mae_sp, cov_sp, preds_sp, trues_sp = evaluate(model_spearman)
    rmse_spsw, mae_spsw, cov_spsw, preds_spsw, trues_spsw = evaluate(model_spearman_sw)



    # ---------- Output Handling ----------
    RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    results_txt_path = os.path.join(RESULTS_DIR, "ubcf_experiment_results.txt")
    results_csv_path = os.path.join(RESULTS_DIR, "ubcf_experiment_results.csv")
    
    output_lines = []
    output_lines.append("=== SIGNIFICANCE WEIGHTING ===")
    output_lines.append(f"RMSE: {rmse_sw}")
    output_lines.append(f"MAE : {mae_sw}")
    output_lines.append(f"Coverage: {cov_sw}")
    
    output_lines.append("\n=== SHRINKAGE PEARSON ===")
    output_lines.append(f"RMSE: {rmse_shr}")
    output_lines.append(f"MAE : {mae_shr}")
    output_lines.append(f"Coverage: {cov_shr}")

    output_lines.append("\n=== COSINE SIMILARITY ===")
    output_lines.append(f"RMSE: {rmse_cos}")
    output_lines.append(f"MAE : {mae_cos}")
    output_lines.append(f"Coverage: {cov_cos}")

    output_lines.append("\n=== SPEARMAN RANK ===")
    output_lines.append(f"RMSE: {rmse_sp}")
    output_lines.append(f"MAE : {mae_sp}")
    output_lines.append(f"Coverage: {cov_sp}")

    output_lines.append("\n=== SPEARMAN SW (Weighted) ===")
    output_lines.append(f"RMSE: {rmse_spsw}")
    output_lines.append(f"MAE : {mae_spsw}")
    output_lines.append(f"Coverage: {cov_spsw}")
    
    # Print to console
    print("\n".join(output_lines))
    
    # Save to TXT
    with open(results_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))
    print(f"\nResults saved to: {results_txt_path}")

    # Save to CSV
    import csv
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Define the new row data
    # Define the new row data conforming to: similarity,params,K_NEIGH,K_PRED,RMSE,MAE,Coverage
    # storing timestamp in params to keep history
    params_str = f"{{'timestamp': '{timestamp}', 'opt': True}}"
    
    new_rows = [
        {"similarity": "Pearson SW (Opt)", "params": params_str, "K_NEIGH": K_NEIGHBORS, "K_PRED": "All", "RMSE": rmse_sw, "MAE": mae_sw, "Coverage": cov_sw},
        {"similarity": "Pearson Shrink (Opt)", "params": params_str, "K_NEIGH": K_NEIGHBORS, "K_PRED": "All", "RMSE": rmse_shr, "MAE": mae_shr, "Coverage": cov_shr},
        {"similarity": "Cosine (K50)", "params": params_str, "K_NEIGH": K_NEIGHBORS, "K_PRED": "All", "RMSE": rmse_cos, "MAE": mae_cos, "Coverage": cov_cos},
        {"similarity": "Spearman (K50)", "params": params_str, "K_NEIGH": K_NEIGHBORS, "K_PRED": "All", "RMSE": rmse_sp, "MAE": mae_sp, "Coverage": cov_sp},
        {"similarity": "Spearman SW (Opt)", "params": params_str, "K_NEIGH": K_NEIGHBORS, "K_PRED": "All", "RMSE": rmse_spsw, "MAE": mae_spsw, "Coverage": cov_spsw},
    ]

    file_exists = os.path.exists(results_csv_path)
    
    with open(results_csv_path, mode="a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        
        if not file_exists:
            writer.writerow(["similarity","params","K_NEIGH","K_PRED","RMSE","MAE","Coverage"])
        
        for row in new_rows:
            writer.writerow([row["similarity"], row["params"], row["K_NEIGH"], row["K_PRED"], row["RMSE"], row["MAE"], row["Coverage"]])
            
    print(f"Results appended to: {results_csv_path}")

    # ---------- Plots ----------
    plt.figure(figsize=(30,5))
    plt.subplot(1,5,1)
    plt.hist(np.array(preds_sw) - np.array(trues_sw), bins=30, alpha=0.7)
    plt.title("Error Dist - Sig. Weighting")

    plt.subplot(1,5,2)
    plt.hist(np.array(preds_shr) - np.array(trues_shr), bins=30, alpha=0.7, color='orange')
    plt.title("Error Dist - Shrinkage")

    plt.subplot(1,5,3)
    plt.hist(np.array(preds_cos) - np.array(trues_cos), bins=30, alpha=0.7, color='green')
    plt.title("Error Dist - Cosine")

    plt.subplot(1,5,4)
    plt.hist(np.array(preds_sp) - np.array(trues_sp), bins=30, alpha=0.7, color='red')
    plt.title("Error Dist - Spearman")
    
    plt.subplot(1,5,5)
    plt.hist(np.array(preds_spsw) - np.array(trues_spsw), bins=30, alpha=0.7, color='purple')
    plt.title("Error Dist - Spearman SW")
    
    plot_path = os.path.join(RESULTS_DIR, "ubcf_experiment_error_dist.png")
    plt.savefig(plot_path)
    print(f"Plot saved to: {plot_path}")
    plt.close()

