"""
Grid Search for UBCF using NDCG@10 on Validation Data.
Modified to fulfill user request: "ubcf_experimentde ndcgye gore result cikaricak grid search yap validation data ile"
"""

import sys
import os
import math
import numpy as np
import pandas as pd
import csv
from datetime import datetime
from functools import partial
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add parent directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from recommender.data_loader import load_all_data
from recommender.UBCF.similarity_user import pearson_sw, pearson_shrink, cosine_sim, spearman_rank, spearman_sw
from recommender.UBCF.neighbors_user import load_or_compute_neighbors
from recommender.UBCF.user_based_cf import UserBasedCF

# Define paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
SPLITS_DIR = os.path.join(PROJECT_ROOT, "data", "splits")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "user_based_cf")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ----------------------------------------------------------------------------
# NDCG Helper Function
# ----------------------------------------------------------------------------
def compute_ndcg_at_k(recommended_items, user_validation_ratings, candidate_pool, k=10, threshold=3.5):
    """
    Compute NDCG@K for a single user with candidate-aware IDCG.
    
    Uses graded relevance (full rating values) and candidate-aware IDCG calculation
    as per Cremonesi et al. (2010) methodology for negative sampling evaluation.
    
    Args:
        recommended_items: List of item_ids recommended by the model (ranked).
        user_validation_ratings: Dict of {item_id: true_rating} for the user in validation set.
        candidate_pool: List of candidate items (relevant + negatives) used for ranking.
        k: Cutoff rank.
        threshold: Rating threshold for relevance (default 3.5 as per evaluation_config).
        
    Returns:
        ndcg_score (float)
        
    References:
        - Järvelin & Kekäläinen (2002): Graded relevance NDCG formula
        - Cremonesi et al. (2010): Candidate-aware IDCG for negative sampling
    """
    # 1. Construct Relevance Vector using GRADED RELEVANCE
    # Graded: Use actual rating values (3.5, 4.0, 4.5, 5.0) for relevant items
    # This better captures the value difference between "liked" and "loved" items
    relevance = []
    
    for mid in recommended_items[:k]:
        true_rating = user_validation_ratings.get(mid, 0.0)
        
        # Graded Relevance: Keep full rating if >= threshold, else 0
        if true_rating >= threshold:
            rel = true_rating  # Full rating (e.g., 4.5, 5.0)
        else:
            rel = 0.0  # Not relevant
            
        relevance.append(rel)
        
    # 2. Compute DCG using graded relevance formula
    # Järvelin & Kekäläinen (2002): DCG = sum((2^rel - 1) / log2(i+2))
    # Exponential gain (2^r - 1) heavily rewards highly-rated items
    def dcg(rel_vec):
        score = 0.0
        for i, r in enumerate(rel_vec):
            if r > 0:
                # Exponential gain: 5-star (2^5-1=31) >> 4-star (2^4-1=15)
                score += (2**r - 1) / np.log2(i + 2)
        return score
        
    actual_dcg = dcg(relevance)
    
    if actual_dcg == 0:
        return 0.0
        
    # 3. Compute CANDIDATE-AWARE IDCG (Critical for negative sampling!)
    # Cremonesi et al. (2010): IDCG should only consider items in the candidate pool
    # Otherwise, NDCG is systematically underestimated when using negative sampling
    
    candidate_set = set(candidate_pool)
    
    # Get relevant ratings ONLY from items in candidate pool
    candidate_relevant_ratings = [
        r for mid, r in user_validation_ratings.items()
        if mid in candidate_set and r >= threshold
    ]
    
    if not candidate_relevant_ratings:
        # No relevant items in candidate pool -> NDCG is 0 (can't do better than 0)
        return 0.0
    
    # Ideal ranking: Sort candidate relevant items by rating (descending)
    ideal_relevance = sorted(candidate_relevant_ratings, reverse=True)[:k]
    ideal_dcg = dcg(ideal_relevance)
    
    if ideal_dcg == 0:
        return 0.0
        
    return actual_dcg / ideal_dcg

# ----------------------------------------------------------------------------
# Evaluation Loop
# ----------------------------------------------------------------------------
def evaluate_model_ndcg(model, R_train, val_df, k=10):
    """
    Evaluates the model on the validation set using NDCG@K.
    """
    ndcg_scores = []
    
    # Group validation data by user for fast lookup
    # val_df has columns: [userId, movieId, rating, ...]
    val_users = val_df['userId'].unique()
    
    # Pre-build validation ratings lookup
    user_val_map = {}
    for uid, group in val_df.groupby('userId'):
        user_val_map[uid] = dict(zip(group['movieId'], group['rating']))
    
    count = 0
    total_users = len(user_val_map)
    
    # print(f"Evaluating on {total_users} users...")
    
    for uid, true_ratings in user_val_map.items():
        if uid not in R_train.index:
            # Skip users not in training set (Cold Start cannot be solved by Pure UBCF)
            continue
        
        # SAFETY: Check if user has valid neighbors in training set
        if uid not in model.neighbors or not model.neighbors[uid]:
            ndcg_scores.append(0.0)
            continue
            
        # 1. Predict scores for all items
        preds = model.predict_for_user(uid)
        
        if not preds:
            ndcg_scores.append(0.0)
            continue
            
        # 2. Filter out items already in training (user has seen them)
        watched_in_train = R_train.loc[uid].dropna().index
        
        # 3. Clean Predictions & SAMPLE CANDIDATES (Scientifically Standard for Offline Eval)
        # Cremonesi et al. (RecSys 2010): "Performance of Recommender Algorithms on Top-N Recommendation Tasks"
        # Standard approach: Rank "Positives" vs "100 Negatives" to avoid full-rank bias.
        # Ranking against 10,000 items (Full Rank) yields tiny scores and is computationally expensive.
        
        # Identify "Relevant" items (Positives) - items rated >= threshold in validation
        relevant_items = [m for m, r in true_ratings.items() if r >= 3.5]
        
        # Get this user's validation items to exclude from negatives
        user_val_items = set(true_ratings.keys())
        
        # Identify Candidates: All items NOT in training and NOT in validation
        all_items = R_train.columns.tolist()
        unwatched_candidates = [m for m in all_items 
                               if m not in watched_in_train 
                               and m not in user_val_items]
        
        # Sample 100 negatives with USER-SPECIFIC SEED for deterministic but varied sampling
        # Cremonesi et al. (2010): Each user should have different but reproducible negative samples
        import random
        random.seed(42 + uid)  # User-specific seed: deterministic but different per user
        if len(unwatched_candidates) > 100:
            negatives = random.sample(unwatched_candidates, 100)
        else:
            negatives = unwatched_candidates
            
        # Final Candidate Pool
        candidate_pool = relevant_items + negatives
        
        # Extract warnings/scores for Candidate Pool ONLY
        pool_preds = []
        for mid in candidate_pool:
            score = preds.get(mid, float('nan'))
            if not np.isnan(score):
                pool_preds.append((mid, score))
            # If model didn't predict (NaN), we treat it as bottom of list (effectively excluded from top K)
            
        if not pool_preds:
            ndcg_scores.append(0.0)
            continue
            
        # 4. Get Top K items from POOL
        top_items = sorted(pool_preds, key=lambda x: x[1], reverse=True)[:k]
        top_item_ids = [m for m, s in top_items]
        
        # DEBUG: Check types for first user
        if count == 0:
            print(f"    [DEBUG] User {uid} Type Check:")
            print(f"       Pred Key Type: {type(top_items[0][0]) if top_items else 'N/A'}")
            print(f"       Val Key Type: {type(list(user_val_map.keys())[0]) if user_val_map else 'N/A'}")
            print(f"       Top 5 Pool IDs: {[x[0] for x in top_items[:5]]}")
            print(f"       Relevant IDs: {relevant_items}")
            # Check overlap
            hits = set(top_item_ids).intersection(set(relevant_items))
            print(f"       Hits in Top {k}: {len(hits)} -> {list(hits)}")
        
        # 5. Compute NDCG with candidate-aware IDCG
        score = compute_ndcg_at_k(top_item_ids, true_ratings, candidate_pool, k=k, threshold=3.5)
        ndcg_scores.append(score)
        
        count += 1
            
    return np.mean(ndcg_scores) if ndcg_scores else 0.0

# ----------------------------------------------------------------------------
# Main Execution
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    print("="*60)
    print("UBCF GRID SEARCH (NDCG@10) - VALIDATION DATA")
    print("="*60)
    
    # 1. Load Data
    print("[1] Loading Data...")
    _, _, _, R_cf, _ = load_all_data()
    
    train_path = os.path.join(SPLITS_DIR, "train.csv")
    val_path = os.path.join(SPLITS_DIR, "validation.csv")
    
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        raise FileNotFoundError("Train or Validation split missing.")
        
    print(f"    Train: {train_path}")
    print(f"    Validation: {val_path}")
    
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    
    train_df["rating"] = train_df["rating"].astype(float)
    val_df["rating"] = val_df["rating"].astype(float)
    
    # Build R_train
    print("    Building R_train matrix...")
    R_train = train_df.pivot(index="userId", columns="movieId", values="rating")
    
    # Pre-calculate means
    user_means = R_train.mean(axis=1)
    item_means = R_train.mean(axis=0)
    global_mean = train_df["rating"].mean()
    
    # 2. Define Grid
    # ADJUSTED: Added '5' to overlap, '50' to neighbors to find a working setting
    K_VALUES = [20, 50]
    OVERLAP_VALUES = [5, 10] # Reduced minimum overlap to solve sparsity
    
    # Similarity Functions configuration
    # We use partials to bind specific params if needed
    # Note: pearson_sw accepts K (significance weighting param), we fix it to 20 or 50.
    
    SIMILARITY_METHODS = [
        ("Pearson_SW", partial(pearson_sw, K=25)), 
        ("Pearson_Shrink", partial(pearson_shrink, LAMBDA=25)),
        ("Cosine", cosine_sim),
        ("Spearman_Rank", spearman_rank),
        ("Spearman_SW", partial(spearman_sw, K=25))
    ]
    
    results = []
    
    print("\n[2] Starting Grid Search...")
    print(f"    Combinations: {len(K_VALUES) * len(OVERLAP_VALUES) * len(SIMILARITY_METHODS)}")
    
    best_ndcg = -1.0
    best_config = None
    
    for sim_name, sim_func in SIMILARITY_METHODS:
        for min_overlap in OVERLAP_VALUES:
            
            current_sim_func = partial(sim_func, MIN_OVERLAP=min_overlap)
            
            for k_neighbors in K_VALUES:
                
                config_name = f"{sim_name}_K{k_neighbors}_Overlap{min_overlap}"
                print(f"\n--- Testing: {config_name} ---")
                
                # 3. Compute/Load Neighbors
                cache_metric_name = f"{sim_name}_Overlap{min_overlap}"
                neighbors = load_or_compute_neighbors(
                    R_train, 
                    current_sim_func, 
                    K=k_neighbors, 
                    metric=cache_metric_name
                )
                
                # 4. Initialize Model
                model = UserBasedCF(
                    R=R_train,
                    neighbors=neighbors,
                    user_means=user_means,
                    item_means=item_means,
                    global_mean=global_mean
                )
                
                # 5. Evaluate (NDCG@10)
                ndcg_score = evaluate_model_ndcg(model, R_train, val_df, k=10)
                
                print(f"    NDCG@10: {ndcg_score:.4f}")
                
                results.append({
                    "Method": sim_name,
                    "K_Neighbors": k_neighbors,
                    "Min_Overlap": min_overlap,
                    "NDCG@10": ndcg_score
                })
                
                if ndcg_score > best_ndcg:
                    best_ndcg = ndcg_score
                    best_config = config_name

    # ----------------------------------------------------------------------------
    # Results saving & Plotting
    # ----------------------------------------------------------------------------
    print("\n" + "="*60)
    print("GRID SEARCH COMPLETED")
    print("="*60)
    print(f"Best Config: {best_config}")
    print(f"Best NDCG@10: {best_ndcg:.4f}")
    
    # Save CSV
    results_csv = os.path.join(RESULTS_DIR, "ubcf_grid_search_ndcg.csv")
    keys = results[0].keys()
    with open(results_csv, 'w', newline='') as f:
        dict_writer = csv.DictWriter(f, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(results)
        
    print(f"Results saved to {results_csv}")
    
    # Sort results for display
    sorted_results = sorted(results, key=lambda x: x['NDCG@10'], reverse=True)
    
    print("\nTop 5 Configurations:")
    for res in sorted_results[:5]:
        print(f"{res['Method']} (K={res['K_Neighbors']}, Overlap={res['Min_Overlap']}): NDCG={res['NDCG@10']:.4f}")

    # Generate Graph
    print("\nGenerating Comparison Graph...")
    try:
        # Aggregate best score per method for plotting
        best_scores_per_method = {}
        for r in results:
            m = r["Method"]
            s = r["NDCG@10"]
            if m not in best_scores_per_method or s > best_scores_per_method[m]:
                best_scores_per_method[m] = s
        
        methods = list(best_scores_per_method.keys())
        scores = list(best_scores_per_method.values())
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(methods, scores, color=['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f1c40f'])
        
        plt.title('UBCF Performance Comparison (Best NDCG@10 per Metric)', fontsize=14)
        plt.xlabel('Similarity Metric', fontsize=12)
        plt.ylabel('NDCG@10 Score', fontsize=12)
        plt.ylim(0, max(scores) * 1.2)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add labels along top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.4f}',
                     ha='center', va='bottom')
        
        output_plot = os.path.join(RESULTS_DIR, "ubcf_ndcg_comparison.png")
        plt.savefig(output_plot, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Graph saved to {output_plot}")
        
    except Exception as e:
        print(f"Error generating graph: {e}")