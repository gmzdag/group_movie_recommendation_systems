
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

from recommender.data_loader import load_movies, load_train_valid_test_splits
from recommender.CBF.content_based import ContentBasedModel


# Define paths relative to the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

def manual_mean_squared_error(y_true, y_pred):
    return np.mean((np.array(y_true) - np.array(y_pred))**2)

def manual_mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))

if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # 1. Load Data
    print("[1] Loading Data and Splits (Temporal)...")
    try:
        train_df, valid_df, test_df = load_train_valid_test_splits()
        movies = load_movies()
    except FileNotFoundError as e:
        print(str(e))
        sys.exit(1)
        
    print(f"Train samples: {len(train_df)}")
    print(f"Valid samples: {len(valid_df)}")
    print(f"Test samples : {len(test_df)}")
    
    # 2. Train Model (Fit)
    # We essentially "feed" the train set into the model as its knowledge base.
    # We combine train and valid for the final evaluation on test, 
    # OR we can rigorously use Train for fitting and Valid for hyperparam tuning (like top_k).
    # For now, let's use Train Only for fitting to see pure generalization from past to future.
    
    print("[2] Initializing Content-Based Model with TRAIN data...")
    # NOTE: ContentBasedModel builds User Profiles from the provided ratings_df
    model = ContentBasedModel(movies, train_df)
    
    # 3. Predict on Test Set
    print("[3] Evaluating on TEST set...")
    preds = []
    trues = []
    
    # Optimization: Iterate efficiently? 
    # Since predict_rating does some work, loop is okay for 16k items.
    
    for i, row in test_df.iterrows():
        uid = row['userId']
        mid = row['movieId']
        actual = row['rating']
        
        # Predict
        # We use default top_k=20 for now. This is a hyperparameter we could tune on Validation set.
        pred = model.predict_rating(uid, mid, top_k=20)
        
        if not np.isnan(pred):
            # Clip
            pred = min(5.0, max(0.5, pred))
            preds.append(pred)
            trues.append(actual)
            
        if (i + 1) % 1000 == 0:
            print(f"Processed {i+1}/{len(test_df)} samples...")
            
    # 4. Metrics
    rmse = math.sqrt(manual_mean_squared_error(trues, preds))
    mae = manual_mean_absolute_error(trues, preds)
    coverage = len(preds) / len(test_df)
    
    # 5. Output Results
    output = []
    output.append("=== CONTENT-BASED FILTERING (CBF) EVALUATION ===")
    output.append("Method: TF-IDF Weighted User Centroid / k-NN Regression")
    output.append(f"Split Strategy: Per-User Temporal Split (Train: {len(train_df)}, Test: {len(test_df)})")
    output.append("-" * 30)
    output.append(f"RMSE    : {rmse:.4f}")
    output.append(f"MAE     : {mae:.4f}")
    output.append(f"Coverage: {coverage:.2%}")
    output.append("-" * 30)
    
    print("\n".join(output))
    
    # Save results
    res_path = os.path.join(RESULTS_DIR, "cbf_temporal_results.txt")
    with open(res_path, "w") as f:
        f.write("\n".join(output))
    print(f"Results saved to {res_path}")
    
    # Plot Error Distribution
    plt.figure(figsize=(6,4))
    plt.hist(np.array(preds) - np.array(trues), bins=30, alpha=0.7, color='purple')
    plt.title(f"CBF Error Distribution (RMSE={rmse:.3f})")
    plt.xlabel("Prediction Error")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    
    plot_path = os.path.join(RESULTS_DIR, "cbf_error_dist.png")
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
    plt.close()
