"""
Hyperparameter Experiment for Item-Based Collaborative Filtering
with ROC & Precision-Recall Analysis

This script:
1. Tests combinations of:
   - normalization: raw, mean_center, zscore
   - similarity: cosine, pearson
   - min_ratings: 3, 5, 10
   - top_k: 10, 20, 40, 60
2. Uses a FIXED Test Set (Train/Test Split) to ensure consistency.
3. Generates ROC and Precision-Recall Curves for the Top 5 configurations.
4. Saves plot to 'itemcf_hyperparam_curves.svg'.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.metrics.pairwise import cosine_similarity

from src.recommender.data_loader import load_ratings, build_cf_matrix

# ------------------------------------------------------------
# Normalization Methods
# ------------------------------------------------------------

def normalize_mean_center(mat):
    """
    Subtracts user mean from ratings. 
    Missing values (NaN) become 0.0 (Neutral) after fillna.
    This corresponds to 'Adjusted Cosine' when used with Cosine Similarity.
    """
    return mat.sub(mat.mean(axis=1), axis=0).fillna(0)

def normalize_zscore(mat):
    mean = mat.mean(axis=1)
    std = mat.std(axis=1).replace(0, 1)
    return mat.sub(mean, axis=0).div(std, axis=0).fillna(0)

normalizers = {
    "mean_center": normalize_mean_center,
    "zscore": normalize_zscore
}

# ------------------------------------------------------------
# Pearson Similarity
# ------------------------------------------------------------
def pearson_sim(mat_T):
    return mat_T.T.corr().fillna(0).values

# ------------------------------------------------------------
#  ItemCF Prediction Formula
# ------------------------------------------------------------
def predict_rating(user_original, user_centered, movie_id, item_sim, k, user_std=1.0):
    """
    user_original: raw ratings (0 = missing)
    user_centered: normalized ratings (centered or zscore)
    user_std: standard deviation of user ratings (default 1.0)
    """

    # If movie not in similarity matrix (filtered out), fallback
    if movie_id not in item_sim.index:
        return user_original[user_original > 0].mean()

    sims = item_sim[movie_id].drop(movie_id)
    top_k = sims.sort_values(ascending=False).head(k)

    neigh_ratings = user_centered[top_k.index]

    # Filter to valid user ratings only (handle NaNs in raw mode)
    valid_mask = neigh_ratings.notna()
    if not valid_mask.any():
        return user_original[user_original > 0].mean()
        
    neigh_ratings = neigh_ratings[valid_mask]
    top_k = top_k[valid_mask]

    # If user has no neighbors ratings, fallback
    if len(neigh_ratings) == 0:
        return user_original[user_original > 0].mean()

    # Weighted sum of neighbors (This effectively predicts the "normalized" value)
    pred_norm = np.dot(top_k.values, neigh_ratings) / np.sum(np.abs(top_k.values))

    # Denormalize
    user_mean = user_original[user_original > 0].mean()
    
    # Correct Reconstruction: Mean + (Deviation * Std)
    return user_mean + (pred_norm * user_std)

# ------------------------------------------------------------
# Evaluation Function
# ------------------------------------------------------------
def evaluate_config(train_df, test_df, norm, sim_type, min_ratings, k):
    """
    Evaluates a single configuration on the provided Train/Test data.
    """
    
    # 1. Filter Train Data by min_ratings
    #    (Standard: Filter based on Train counts)
    counts = train_df["movieId"].value_counts()
    valid_ids = counts[counts >= min_ratings].index
    
    # If filter removes too much, we just proceed with what we have
    train_filtered = train_df[train_df["movieId"].isin(valid_ids)]
    
    # Build Matrices
    raw_um = build_cf_matrix(train_filtered)
    
    # Check emptiness
    if raw_um.empty:
        return np.nan, np.nan, [], []

    # Normalize
    norm_um = normalizers[norm](raw_um)
    
    # Pre-compute STDs if needed
    if norm == "zscore":
        user_stds = raw_um.std(axis=1).replace(0, 1)
    else:
        user_stds = pd.Series(1.0, index=raw_um.index)

    # Similarity matrix
    if sim_type == "cosine":
        # Cosine requires dense input (0 for missing)
        item_sim = pd.DataFrame(
            cosine_similarity(norm_um.fillna(0).T),
            index=norm_um.columns,
            columns=norm_um.columns
        )
    else:
        item_sim = pd.DataFrame(
            pearson_sim(norm_um.T),
            index=norm_um.columns,
            columns=norm_um.columns
        )

    # Prediction Loop
    y_true = []
    y_pred = []

    # Valid test set: User must exist in Train, Movie must exist in Train (and be in valid_ids)
    valid_users = set(raw_um.index)
    valid_movies = set(raw_um.columns)
    
    for _, row in test_df.iterrows():
        user = row["userId"]
        movie = row["movieId"]
        true_rating = row["rating"]

        if user not in valid_users or movie not in valid_movies:
            # Cannot predict for cold-start in this pure ItemCF setup
            continue

        user_orig = raw_um.loc[user]
        user_norm = norm_um.loc[user]
        std_val = user_stds.loc[user]

        try:
            pred = predict_rating(user_orig, user_norm, movie, item_sim, k, user_std=std_val)
            
            if np.isnan(pred) or np.isinf(pred):
                continue
                
            y_true.append(true_rating)
            y_pred.append(pred)
        except Exception:
            continue

    if not y_true:
        return np.nan, np.nan, [], []

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    return mae, rmse, y_true, y_pred

# ------------------------------------------------------------
# Plotting
# ------------------------------------------------------------
def plot_results(results, output_file="itemcf_hyperparam_curves.svg"):
    # Sort by RMSE (ascending) and take Top 5
    top_5 = sorted(results, key=lambda x: x["RMSE"])[:5]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot 1: ROC Curve
    ax_roc = axes[0]
    ax_roc.plot([0, 1], [0, 1], "k--", label="Random")
    
    # Plot 2: Precision-Recall
    ax_pr = axes[1]
    
    print("\n--- Summary Table ---")
    print(f"{'Config':<50} | {'ROC_AUC':<8} | {'AP':<8}")
    print("-" * 75)

    for res in top_5:
        y_true = np.array(res["y_true"])
        y_score = np.array(res["y_pred"])
        
        # Binarize y_true (Threshold = 3.5)
        y_true_binary = (y_true >= 3.5).astype(int)
        
        # Check if single class
        if len(np.unique(y_true_binary)) < 2:
            print(f"Skipping {res['id']}: Only one class in test set.")
            continue
            
        # ROC
        fpr, tpr, _ = roc_curve(y_true_binary, y_score)
        roc_auc = auc(fpr, tpr)
        
        # Precision-Recall
        precision, recall, _ = precision_recall_curve(y_true_binary, y_score)
        avg_prec = average_precision_score(y_true_binary, y_score)
        
        label = f"{res['normalization']}/{res['similarity']} (k={res['top_k']})"
        
        ax_roc.plot(fpr, tpr, label=f"{label} (AUC={roc_auc:.3f})")
        ax_pr.plot(recall, precision, label=f"{label} (AP={avg_prec:.3f})")
        
        # Console output
        config_str = f"{res['normalization']} | {res['similarity']} | min={res['min_ratings']} | k={res['top_k']}"
        print(f"{config_str:<50} | {roc_auc:.4f}   | {avg_prec:.4f}")

    # Styling ROC
    ax_roc.set_title("ROC Curve (Top 5 Configs)")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.legend(loc="lower right")
    ax_roc.grid(True)
    
    # Styling PR
    ax_pr.set_title("Precision-Recall Curve (Top 5 Configs)")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.legend(loc="lower left")
    ax_pr.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_file, format="svg", bbox_inches="tight")
    print("-" * 75)
    print(f"\nSaved plot to {output_file}")


# ------------------------------------------------------------
# MAIN: Grid Search
# ------------------------------------------------------------
if __name__ == "__main__":
    
    # 1. Load Data
    all_ratings = load_ratings()
    
    # 2. Fixed Train/Test Split
    # Sample 1000 interactions for testing
    test_df = all_ratings.sample(n=1000, random_state=42)
    train_df = all_ratings.drop(test_df.index)
    
    print(f"Data Split: Train={len(train_df)}, Test={len(test_df)}")
    
    results = []
    
    # Grid Search
    norms = ["mean_center", "zscore"]
    sims = ["cosine", "pearson"]
    min_ratings_list = [3, 5, 10]
    k_list = [10, 20, 40, 60]
    
    total_configs = len(norms) * len(sims) * len(min_ratings_list) * len(k_list)
    idx = 0
    
    print(f"Starting Grid Search ({total_configs} configurations)...")

    for norm in norms:
        for sim in sims:
            for min_r in min_ratings_list:
                for k in k_list:
                    idx += 1
                    try:
                        mae, rmse, y_t, y_p = evaluate_config(
                            train_df, test_df, norm, sim, min_r, k
                        )
                        
                        if np.isnan(rmse):
                            print(f"[{idx}/{total_configs}] Skipped: {norm}+{sim}, min={min_r}, k={k} (No predictions)")
                            continue
                            
                        print(f"[{idx}/{total_configs}] {norm} + {sim}, min={min_r}, k={k} → RMSE={rmse:.4f}")

                        results.append({
                            "id": f"{norm}_{sim}_{min_r}_{k}",
                            "normalization": norm,
                            "similarity": sim,
                            "min_ratings": min_r,
                            "top_k": k,
                            "MAE": mae,
                            "RMSE": rmse,
                            "y_true": y_t,
                            "y_pred": y_p
                        })
                        
                    except Exception as e:
                        print(f"Error in config {norm}+{sim}: {e}")

    # Plot
    df = pd.DataFrame([{k:v for k,v in r.items() if k not in ['y_true', 'y_pred']} for r in results])
    df.to_csv("itemcf_results.csv", index=False)
    print("\nSaved → itemcf_results.csv")
    
    if results:
        plot_results(results)
    else:
        print("No valid results to plot.")
