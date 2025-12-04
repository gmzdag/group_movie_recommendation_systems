"""
Item-Based Similarity Functions
-------------------------------
Computes item–item similarity matrices.
"""

import pandas as pd
import numpy as np


def pearson_similarity(norm_um: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Pearson similarity between items (movies).
    
    Args:
        norm_um: U × M matrix (users × movies) - NORMALIZED ratings
    
    Returns:
        similarity_df: M × M similarity matrix (movies × movies)
    """
    print(f"[DEBUG] pearson_similarity() - Input shape: {norm_um.shape}")
    print(f"[DEBUG] Computing correlation between COLUMNS (movies)...")
    
    sim = norm_um.corr()
    
    print(f"[DEBUG] Similarity matrix shape: {sim.shape}")
    print(f"[DEBUG] Sample similarity values (first movie):")
    print(f"        - Min: {sim.iloc[0].min():.4f}")
    print(f"        - Max: {sim.iloc[0].max():.4f}")
    print(f"        - Mean: {sim.iloc[0].mean():.4f}")
    
    # NaN → 0 similarity
    sim_filled = sim.fillna(0)
    nan_count = sim.isna().sum().sum()
    print(f"[DEBUG] Filled {nan_count} NaN values with 0")
    
    return sim_filled