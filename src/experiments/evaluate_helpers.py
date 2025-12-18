
import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score
from src.recommender.data_splitter import temporal_train_validation_test_split

def get_temporal_split(ratings_df):
    """
    Wrapper for temporal split.
    Returns: train, val, test
    """
    return temporal_train_validation_test_split(ratings_df, train_ratio=0.7, valid_ratio=0.15)

def calculate_ndcg(model, test_df, k=10):
    """
    Calculates average NDCG@K for a model on test_df.
    Handles models returning (score, info) tuples.
    """
    ndcg_scores = []
    
    # Group by user
    # Optimization: For speed, maybe sample users?
    # Full eval on validation set
    
    grouped = test_df.groupby('userId')
    
    for uid, group in grouped:
        if len(group) < 2: continue
        
        true_ratings = group['rating'].values
        true_mids = group['movieId'].values
        
        # Predict scores
        pred_scores = []
        for mid in true_mids:
            try:
                # Handle Model Variance
                res = model.predict(uid, mid)
                
                # Unwrap tuple if needed (Model 2 returns (score, source))
                if isinstance(res, tuple):
                    score = res[0]
                else:
                    score = res
                    
                if np.isnan(score):
                    score = 0
            except:
                score = 0
                
            pred_scores.append(score)
            
        # Calc NDCG
        # sklearn requires shape (n_samples, n_items)
        if len(pred_scores) > 1:
            score = ndcg_score([true_ratings], [pred_scores], k=k)
            ndcg_scores.append(score)
            
    return np.mean(ndcg_scores) if ndcg_scores else 0.0
