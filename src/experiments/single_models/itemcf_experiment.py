"""
Hyperparameter Experiment for Item-Based Collaborative Filtering
Audited and Refactored for Methodological Correctness (IEEE Standards).

CHANGELOG (Final Production Version):
1. FIXED: Positive items correctly filtered by relevance threshold (rating >= 3.5)
2. FIXED: User-specific deterministic negative sampling (seed + user_id)
3. FIXED: Pearson similarity uses pairwise complete observations
4. FIXED: Rated items inferred from raw matrix NaNs (not normalized values)
5. FIXED: Negative pool excludes both rated AND positive items (no duplicates)
6. FIXED: Pearson neighbor selection uses absolute similarity magnitude
7. FIXED: Deterministic candidate ordering (sorted lists)
8. DOCUMENTED: Candidate universe restricted to training-visible items

Assumptions:
- Binary Relevance: rating >= 3.5 = Relevant (1), else Irrelevant (0)
- Negative Sampling: 100 random unrated items per user
- Candidate Universe: Items observed in training after min_ratings filtering
- Imputation: Global Mean for unpredictable items
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ndcg_score
from sklearn.metrics.pairwise import cosine_similarity
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.recommender.data_loader import build_cf_matrix, load_train_valid_test_splits

plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14
})

RELEVANCE_THRESHOLD = 3.5
NEGATIVE_SAMPLES = 100
RANDOM_SEED = 42

def normalize_mean_center(mat):
    # SCIENTIFIC FIX: Do NOT fillna(0) yet. Keep NaNs for correct correlation.
    return mat.sub(mat.mean(axis=1), axis=0)

def normalize_zscore(mat):
    mean = mat.mean(axis=1)
    std = mat.std(axis=1)
    
    # SCIENTIFIC FIX: Users with 0 std dev cannot be normalized (z-score undefined).
    # Setting std to 1 implies variance exists where there is none.
    # We set them to NaN to exclude from calculation, or keep as NaN.
    # We use replace(0, np.nan) to ensure we don't divide by zero.
    std = std.replace(0, np.nan)
    
    return mat.sub(mean, axis=0).div(std, axis=0)

normalizers = {
    "mean_center": normalize_mean_center,
    "zscore": normalize_zscore
}

def calculate_similarity(train_df, norm_method, sim_method, min_ratings=5):
    # 1. Build Matrix with NaNs (Sparse)
    raw_um = build_cf_matrix(train_df)
    
    # 2. Normalize (User-Based handling of Bias) - Preserving NaNs
    norm_um = normalizers[norm_method](raw_um)
    
    # 3. Compute Item Similarity
    if sim_method == "cosine":
        # Cosine requires 0-filled for dot product, BUT we must be careful.
        # Adjusted Cosine: We want Cosine on the adjusted vectors.
        # Filling NaNs with 0 here is strictly correct for "Adjusted Cosine" 
        # because 0 implies "Average Rating" after centering.
        norm_um_filled = norm_um.fillna(0)
        item_sim = pd.DataFrame(
            cosine_similarity(norm_um_filled.T),
            index=norm_um.columns,
            columns=norm_um.columns
        )
    elif sim_method == "pearson":
        # SCIENTIFIC FIX: Pearson strictly requires NaNs to ignore missing pairs.
        # We MUST NOT fillna(0) here. 
        # Also added min_periods to avoid spurious correlations (Variance Collapse).
        # FIX: Compute correlation on columns (Items), not rows (Users). removed .T
        item_sim = norm_um.corr(method="pearson", min_periods=min_ratings).fillna(0)
    else:
        raise ValueError(f"Unknown similarity: {sim_method}")
        
    # Final cleanup: fill diagonal with 0
    np.fill_diagonal(item_sim.values, 0)
        
    # Return raw_um (with NaNs) for prediction checks, norm_um (filled) for vector ops
    return raw_um, norm_um.fillna(0), item_sim

def predict_rating_item_based(user_normalized_vector, item_vector, movie_id, n_neighbors, rated_items, sim_method="cosine"):
    """
    Predicts rating using Item-Based CF logic.
    SCIENTIFIC FIX: Explicitly passes 'rated_items' logic.
    Previously, checking (user_vector != 0) caused ratings equal to the mean (normalized to 0) to be ignored.
    Now, even if normalized rating is 0.0, it is included in the weighted sum if it exists in rated_items.
    """
    # 1. Intersection: Items user rated AND that have similarity with target movie
    # valid_items are neighbors (j) that user has rated (r_uj exists)
    valid_items = [item for item in rated_items if item != movie_id and item in item_vector.index]
        
    if len(valid_items) == 0:
        return np.nan 
        
    # Vector of similarities between target(i) and neighbors(j)
    sims = item_vector.loc[valid_items]
    
    # Vector of user's normalized ratings for neighbors(j)
    # Note: Values can be 0.0 (if rating == mean), this is valid information.
    user_ratings_for_sims = user_normalized_vector.loc[valid_items]
    
    # FIX: For Pearson, select neighbors by absolute similarity magnitude
    # Strong negative correlations are as informative as positive ones
    if sim_method == "pearson":
        # Sort by absolute value, take top K
        top_k_indices = sims.abs().nlargest(n_neighbors).index
        top_k_sims = sims.loc[top_k_indices]
        top_k_ratings = user_ratings_for_sims.loc[top_k_indices]
    else:
        # Sort by actual value (Cosine), take top K
        top_k_indices = sims.nlargest(n_neighbors).index
        top_k_sims = sims.loc[top_k_indices]
        top_k_ratings = user_ratings_for_sims.loc[top_k_indices]
    
    # SCIENTIFIC FIX: For Pearson, sum can be 0 due to +/- cancellation, but info exists
    # Check if ALL similarities are actually zero (magnitude check)
    if np.allclose(top_k_sims.abs().values, 0) or top_k_sims.abs().sum() == 0:
        return np.nan
        
    # Weighted Sum Formula: Sum(sim_ij * r_uj) / Sum(|sim_ij|)
    numerator = np.dot(top_k_sims.values, top_k_ratings.values)
    denominator = np.sum(np.abs(top_k_sims.values))
    
    if denominator == 0:
        return np.nan
        
    return numerator / denominator


def evaluate_model(train_df, eval_df, config, n_negative_samples=NEGATIVE_SAMPLES, random_seed=RANDOM_SEED):
    """
    Evaluates ItemCF with Negative Sampling for realistic NDCG.
    
    FIXES APPLIED:
    1. Positive items = eval items with rating >= RELEVANCE_THRESHOLD
    2. User-specific deterministic negative sampling
    3. Candidate universe = training-visible items only
    4. Deterministic candidate ordering
    """
    norm, sim, k, min_r = config['normalization'], config['similarity'], config['top_k'], config['min_ratings']
    
    counts = train_df["movieId"].value_counts()
    valid_ids = counts[counts >= min_r].index
    train_filtered = train_df[train_df["movieId"].isin(valid_ids)]
    
    if train_filtered.empty:
        return 0.0
        
    raw_um, norm_um, item_sim = calculate_similarity(train_filtered, norm, sim, min_ratings=5) \
                                if sim == "pearson" else calculate_similarity(train_filtered, norm, sim)
    global_mean = train_filtered["rating"].mean()
    
    # CANDIDATE UNIVERSE: Restricted to items observed in training after min_ratings filtering
    # This is standard practice in CF evaluation to avoid cold-start items in ranking
    all_items = set(raw_um.columns)
    
    if norm == "zscore":
        user_means = raw_um.mean(axis=1)
        user_stds = raw_um.std(axis=1).replace(0, 1)
    elif norm == "mean_center":
        user_means = raw_um.mean(axis=1)
        user_stds = None
    else:
        user_means = None
        user_stds = None
        
    eval_users_grouped = eval_df.groupby("userId")
    ndcg_scores = []
    
    # Track statistics for RMSE/MAE on positive items
    all_true_ratings = []
    all_pred_ratings = []
    
    for user_id, user_data in eval_users_grouped:
        
        # FIX: Positive items = only those with rating >= RELEVANCE_THRESHOLD
        positive_mask = user_data["rating"] >= RELEVANCE_THRESHOLD
        positive_data = user_data[positive_mask]
        
        # FIX: Skip users with zero positive items (cannot compute NDCG)
        if len(positive_data) == 0:
            continue
        
        if user_id in raw_um.index:
            user_train_ratings = norm_um.loc[user_id]
            # FIX: Infer rated items from raw matrix (NaN = not rated)
            user_rated_items_train = set(raw_um.loc[user_id].dropna().index)
        else:
            user_train_ratings = None
            user_rated_items_train = set()
        
        # SCIENTIFIC FIX: Include eval set items in rated_items for proper negative sampling
        # "Negative" should mean "truly unseen", not "seen in eval but not in train"
        user_rated_items_eval = set(user_data["movieId"].values)
        user_rated_items = user_rated_items_train | user_rated_items_eval
        
        positive_items = set(positive_data["movieId"].values)
        
        # FIX: Negative pool = all items EXCEPT rated items (train + eval)
        # This ensures negatives are TRULY unseen movies
        candidate_negatives = all_items - user_rated_items
        
        # FIX: User-specific deterministic sampling
        # Each user gets the same negatives regardless of iteration order
        user_rng = np.random.default_rng(seed=random_seed + int(user_id))
        
        if len(candidate_negatives) > n_negative_samples:
            negative_items = user_rng.choice(list(candidate_negatives), n_negative_samples, replace=False)
        else:
            negative_items = list(candidate_negatives)
        
        # FIX: Deterministic candidate ordering
        all_candidate_items = sorted(positive_items) + sorted(negative_items)
        
        item_scores = []
        item_relevance = []
        
        for movie_id in all_candidate_items:
            # Relevance: 1 if in positive_items, 0 otherwise
            relevance = 1 if movie_id in positive_items else 0
            
            pred = np.nan
            if user_train_ratings is not None and movie_id in item_sim.index:
                # CRITICAL: Use only TRAIN rated items for neighbor selection (avoid data leakage)
                pred_norm = predict_rating_item_based(user_train_ratings, item_sim[movie_id], movie_id, k, user_rated_items_train, sim_method=sim)
                
                if not np.isnan(pred_norm):
                    if norm == "zscore":
                        pred = user_means[user_id] + (pred_norm * user_stds[user_id])
                    elif norm == "mean_center":
                        pred = user_means[user_id] + pred_norm
            
            if np.isnan(pred):
                pred = global_mean
            
            pred = min(max(pred, 0.5), 5.0)
            
            item_scores.append(pred)
            item_relevance.append(relevance)
        
        # Compute NDCG@10
        try:
            score = ndcg_score([item_relevance], [item_scores], k=10)
            ndcg_scores.append(score)
        except ValueError:
            pass
            
        # Accumulate strictly true positive ratings for RMSE/MAE
        if positive_data.empty: continue
        
        # For RMSE, we need: True Rating vs Predicted Rating for the known positive items
        for idx, row in positive_data.iterrows():
            mid = row['movieId']
            true_r = row['rating']
            
            # Re-predict specifically for accuracy metrics
            pred_r = np.nan
            if user_train_ratings is not None and mid in item_sim.index:
                p_norm = predict_rating_item_based(user_train_ratings, item_sim[mid], mid, k, user_rated_items_train, sim_method=sim)
                if not np.isnan(p_norm):
                    if norm == "zscore":
                        pred_r = user_means[user_id] + (p_norm * user_stds[user_id])
                    elif norm == "mean_center":
                        pred_r = user_means[user_id] + p_norm
            
            if np.isnan(pred_r):
                pred_r = global_mean
            
            pred_r = min(max(pred_r, 0.5), 5.0)
            
            all_true_ratings.append(true_r)
            all_pred_ratings.append(pred_r)
    
    if not ndcg_scores:
        return {'NDCG': 0.0, 'RMSE': np.nan, 'MAE': np.nan}
    
    avg_ndcg = float(np.mean(ndcg_scores))
    
    # Calculate global RMSE/MAE for this config on positives
    if all_pred_ratings:
        mse = np.mean(np.square(np.array(all_true_ratings) - np.array(all_pred_ratings)))
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(np.array(all_true_ratings) - np.array(all_pred_ratings)))
    else:
        rmse = np.nan
        mae = np.nan
        
    return {'NDCG': avg_ndcg, 'RMSE': rmse, 'MAE': mae}

def generate_visualizations(results_df, best_config, val_metrics, test_metrics, output_dir):
    results_sorted = results_df.sort_values(by="NDCG", ascending=False)
    results_sorted.to_csv(os.path.join(output_dir, "validation_results_table.csv"), index=False)
    results_sorted.to_csv(os.path.join(output_dir, "itemcf_results.csv"), index=False)
    
    top_10 = results_sorted.head(10).copy()
    top_10["label"] = top_10.apply(lambda x: f"{x['normalization'][:1]}+{x['similarity'][:3]}, m={x['min_ratings']}, k={x['top_k']}", axis=1)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=top_10, x="label", y="NDCG", color="#4682B4")
    plt.xticks(rotation=45, ha='right')
    plt.title("Top-10 ItemCF Configurations (Validation Set)")
    plt.ylabel("NDCG@10")
    plt.xlabel("Configuration")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig_1_top10_validation.svg"))
    plt.close()
    
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=results_df, x="top_k", y="NDCG", hue="normalization", style="min_ratings", markers=True, dashes=False)
    plt.title("Effect of Neighborhood Size (K) on NDCG@10")
    plt.ylabel("Validation NDCG@10")
    plt.xlabel("Top K Neighbors")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig_2_sensitivity_k.svg"))
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.lineplot(data=results_df, x="min_ratings", y="NDCG", hue="normalization", style="top_k", markers=True, dashes=False)
    plt.title("Effect of Min Ratings Threshold on NDCG@10")
    plt.ylabel("Validation NDCG@10")
    plt.xlabel("Min Ratings Filter")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig_3_sensitivity_min_ratings.svg"))
    plt.close()
    
    comparison_df = pd.DataFrame({
        "Split": ["Validation", "Test"],
        "NDCG@10": [val_metrics['NDCG'], test_metrics['NDCG']]
    })
    
    plt.figure(figsize=(6, 6))
    ax = sns.barplot(data=comparison_df, x="Split", y="NDCG@10", palette=["#A9A9A9", "#228B22"])
    plt.title(f"Generalization Check\n(Best Config: {best_config['normalization']}+{best_config['similarity']})")
    plt.ylim(0, 1.0)
    for i, v in enumerate([val_metrics['NDCG'], test_metrics['NDCG']]):
        ax.text(i, v + 0.01, f"{v:.4f}", ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig_4_val_vs_test.svg"))
    plt.close()

if __name__ == "__main__":
    
    print("--- Loading Data ---")
    train_df, valid_df, test_df = load_train_valid_test_splits()
    print(f"Train: {len(train_df)}, Valid: {len(valid_df)}, Test: {len(test_df)}")
    
    valid_pairings = [
        ("mean_center", "cosine"),
        ("zscore", "cosine"),
        ("mean_center", "pearson")
    ]
    min_ratings_options = [3, 5, 10, 15, 20]
    k_options = [10, 20, 40, 60]
    
    results = []
    
    print("\n--- Starting Grid Search on VALIDATION SET ---")
    print(f"{'Config':<50} | {'NDCG@10':<8}")
    print("-" * 65)
    
    best_config = None
    best_score = -1.0
    
    for norm, sim in valid_pairings:
        for min_r in min_ratings_options:
            for k in k_options:
                config = {
                    "normalization": norm,
                    "similarity": sim,
                    "min_ratings": min_r,
                    "top_k": k
                }
                
                try:
                    val_metrics = evaluate_model(train_df, valid_df, config)
                    ndcg = val_metrics['NDCG']
                    config_str = f"{norm}+{sim}, min={min_r}, k={k}"
                    
                    print(f"{config_str:<50} | {ndcg:.4f} (RMSE: {val_metrics['RMSE']:.4f})")
                    
                    # Merge config with metrics for saving
                    result_entry = {**config}
                    result_entry.update(val_metrics)
                    results.append(result_entry)
                    
                    if ndcg > best_score:
                        best_score = ndcg
                        best_config = config
                        
                except Exception as e:
                    print(f"Failed config {config}: {e}")
                    import traceback
                    traceback.print_exc()
                    
    print("-" * 65)
    print(f"Best Config Found: {best_config}")
    print(f"Best Validation NDCG: {best_score:.4f}")
    
    print("\n--- FINAL EVALUATION on TEST SET ---")
    if best_config:
        test_metrics = evaluate_model(train_df, test_df, best_config)
        test_ndcg = test_metrics['NDCG']
        print(f"FINAL TEST NDCG@10: {test_ndcg:.4f}")
        
        RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
        os.makedirs(RESULTS_DIR, exist_ok=True)
        
        with open(os.path.join(RESULTS_DIR, "itemcf_final_report.txt"), "w") as f:
            f.write("--- ItemCF Final Experiment Report ---\n")
            f.write(f"Best Configuration:\n{best_config}\n\n")
            f.write(f"Validation NDCG: {best_score:.4f}\n")
            f.write(f"Test NDCG: {test_ndcg:.4f}\n")
            f.write(f"Generalization Gap: {abs(best_score - test_ndcg):.4f}\n")
        
        results_df = pd.DataFrame(results)
        generate_visualizations(results_df, best_config, {'NDCG': best_score}, test_metrics, RESULTS_DIR)
        print(f"\nArtifacts generated in {RESULTS_DIR}")
    else:
        print("No valid configuration found.")
