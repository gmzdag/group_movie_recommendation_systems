
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ndcg_score
from sklearn.metrics.pairwise import cosine_similarity
from src.recommender.data_loader import load_movies, build_cf_matrix, load_train_valid_test_splits
from src.recommender.IBCF.item_based_cf import ItemBasedCF
from src.recommender.CBF.content_based import ContentBasedModel
from src.recommender.IBCF.neighbors_item import compute_item_neighbors
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

def hybrid_predict(user_id, movie_id, n_neighbors, ib_pred, cb_pred, C):
    """
    Hybrid Formula:
    Score = (n / (n + C)) * IB_Score + (C / (n + C)) * CB_Score
    """
    # If IB is NaN, fallback to CB completely
    if np.isnan(ib_pred):
        return cb_pred
    
    # If CB is NaN, fallback to IB (rare, but possible if no history logic)
    if np.isnan(cb_pred):
        return ib_pred
        
    alpha = n_neighbors / (n_neighbors + C)
    beta = C / (n_neighbors + C)
    
    return (alpha * ib_pred) + (beta * cb_pred)

def evaluate_c(test_df, ibcf, cb, c_values=[0, 1, 5, 10, 20, 50]):
    """
    Evaluates different C values using NDCG.
    """
    results = {c: [] for c in c_values}
    
    print(f"\n[EVAL] Starting Evaluation on {len(test_df)} interactions...")
    
    # Group by User for NDCG calculation
    grouped = test_df.groupby('userId')
    
    ndcg_scores = {c: [] for c in c_values}
    
    processed_users = 0
    
    for uid, group in grouped:
        if len(group) < 2:
            continue # NDCG needs >1 item to be meaningful usually (or at least list)
            
        true_ratings = group['rating'].values
        
        # Prepare arrays
        ib_preds = []
        cb_preds = []
        ns = []
        
        valid_indices = []
        
        for idx, row in group.iterrows():
            mid = row['movieId']
            
            # IB Prediction
            ib_p, info = ibcf.predict(uid, mid, return_info=True)
            n = info['n_neighbors']
            
            # CB Prediction
            cb_p = cb.predict_rating(uid, mid)
            
            ib_preds.append(ib_p)
            cb_preds.append(cb_p)
            ns.append(n)
            valid_indices.append(idx)
            
        # Optimization Loop
        for c in c_values:
            y_pred = []
            y_true_clean = []
            
            for i in range(len(valid_indices)):
                ib = ib_preds[i]
                cb_val = cb_preds[i]
                n = ns[i]
                
                # Hybrid Logic
                final = hybrid_predict(uid, 0, n, ib, cb_val, c) # mid irrelevant for formula
                
                if not np.isnan(final):
                    y_pred.append(final)
                    y_true_clean.append(true_ratings[i])
            
            if len(y_pred) > 1:
                # Calculate NDCG
                # sklearn ndcg_score expects (n_samples, n_items)
                score = ndcg_score([y_true_clean], [y_pred])
                ndcg_scores[c].append(score)
        
        processed_users += 1
        if processed_users % 50 == 0:
            print(f"Processed {processed_users} users...")

    # Aggregating
    final_scores = {}
    for c in c_values:
        if ndcg_scores[c]:
            final_scores[c] = np.mean(ndcg_scores[c])
        else:
            final_scores[c] = 0.0
            
    return final_scores

def normalize_zscore(mat):
    mean = mat.mean(axis=1)
    std = mat.std(axis=1).replace(0, 1)
    return mat.sub(mean, axis=0).div(std, axis=0).fillna(0)

def main():
    # 1. Load fixed splits
    print("Loading Fixed Splits...")
    train_df, valid_df, test_df = load_train_valid_test_splits()
    movies = load_movies()
    
    print(f"Train: {len(train_df)}, Valid: {len(valid_df)}, Test: {len(test_df)}")
    
    # Note: We use VALIDATION set for C optimization (hyperparameter tuning)
    # Test set should be reserved for final evaluation
    eval_df = valid_df
    
    # 3. Setup Models
    
    # --- IBCF Setup ---
    print("Building IBCF Model...")
    raw_um = build_cf_matrix(train_df)
    norm_um = normalize_zscore(raw_um)
    
    # Compute Sim
    print("Computing Cosine Similarity for IBCF...")
    sim_matrix = cosine_similarity(norm_um.fillna(0).T)
    sim_df = pd.DataFrame(sim_matrix, index=norm_um.columns, columns=norm_um.columns)
    
    neighbor_sims = compute_item_neighbors(sim_df, K=20) # k=20 fixed for this exp
    
    ibcf = ItemBasedCF(raw_um, norm_um, neighbor_sims, movies, top_k=20)
    
    # --- CB Setup ---
    print("Building Content-Based Model...")
    cb = ContentBasedModel(movies, train_df)
    
    # 4. Evaluate C on Validation Set
    c_candidates = [0, 1, 2, 5, 10, 15, 20, 30, 50, 100]
    scores = evaluate_c(eval_df, ibcf, cb, c_candidates)
    
    print("\n--- Results (NDCG) ---")
    best_c = None
    best_score = -1
    
    for c, s in scores.items():
        print(f"C={c}: {s:.4f}")
        if s > best_score:
            best_score = s
            best_c = c
            
    print(f"\nBest C: {best_c} (NDCG: {best_score:.4f})")
    
    # 5. Plot
    plt.figure(figsize=(10, 6))
    plt.plot(list(scores.keys()), list(scores.values()), marker='o')
    plt.title(f"Hybrid Parameter Optimization (0-5 Range) (Best C={best_c})")
    plt.xlabel("C Parameter (Trust Factor)")
    plt.ylabel("Average NDCG")
    plt.grid(True)
    
    # Save
    os.makedirs("results/graphs", exist_ok=True)
    
    png_path = "results/graphs/hybrid_c_optimization_fine.png"
    svg_path = "results/graphs/hybrid_c_optimization_fine.svg"
    
    plt.savefig(png_path)
    plt.savefig(svg_path)
    
    print(f"Saved plots to:\n - {png_path}\n - {svg_path}")

if __name__ == "__main__":
    main()
