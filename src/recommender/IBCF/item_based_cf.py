"""
Item-Based Collaborative Filtering (Signal Provider)
----------------------------------------------------
Optimized for Hybrid System Integration.
- Normalization: Z-SCORE
- Similarity: COSINE
- Prediction: mean + (weighted_sum * std)
- Output: Single prediction score (or NaN)
"""

import numpy as np
import pandas as pd


class ItemBasedCF:
    def __init__(self, raw_um, norm_um, item_neighbors, movies, top_k=10):
        """
        Args:
            raw_um: Raw Rating Matrix (Users x Movies), Missing=NaN
            norm_um: Z-Score Normalized Matrix, Missing=0
            item_neighbors: Pre-computed top-K neighbors dict
            movies: Movies metadata (reference)
            top_k: Neighbor count (Fixed=10 per request)
        """
        print(f"\n[DEBUG] Initializing ItemBasedCF (Hybrid Signal Mode)...")
        print(f"[DEBUG] - raw_um shape: {raw_um.shape}")
        print(f"[DEBUG] - norm_um shape: {norm_um.shape}")
        print(f"[DEBUG] - top_k: {top_k}")
        
        self.raw_um = raw_um
        self.norm_um = norm_um
        self.item_neighbors = item_neighbors
        self.movies = movies
        self.top_k = top_k
        
        # ------------------------------------------------------------------
        # SAFETY CHECK: Ensure norm_um is Z-Score Normalized
        # ------------------------------------------------------------------
        # Check 1: Mean is approx 0
        row_means = self.norm_um.mean(axis=1)
        global_mean = row_means.mean()
        
        # Check 2: Std of non-zero values (approx check)
        # We can't easily check 'std=1' on sparse 0-filled data without re-masking,
        # but we can assume if mean is 0 it's likely centered/standardized.
        
        if abs(global_mean) > 0.1:
            print(f"\n[WARNING] 'norm_um' does NOT appear to be Z-Score Normalized!")
            print(f"          Global mean: {global_mean:.4f} (Expected ~0.0)")
            print(f"          Prediction formula requires Z-Score input.")

        # ------------------------------------------------------------------
        # Pre-compute User Statistcs (Mean & Std)
        # ------------------------------------------------------------------
        print(f"[DEBUG] Pre-computing user statistics (Mean & Std)...")
        
        # Mean (ignoring NaNs)
        self.user_means = self.raw_um.mean(axis=1)
        
        # Standard Deviation (ignoring NaNs)
        # Replace 0 std with 1.0 to avoid multiplication issues
        self.user_stds = self.raw_um.std(axis=1).fillna(1.0).replace(0, 1.0)
        
        print(f"[DEBUG] User Means range: [{self.user_means.min():.2f}, {self.user_means.max():.2f}]")
        print(f"[DEBUG] User Stds range:  [{self.user_stds.min():.2f}, {self.user_stds.max():.2f}]")


    # ------------------------------------------------------------
    # Predict rating (Signal Only)
    # ------------------------------------------------------------
    def predict(self, user_id, movie_id, verbose=False):
        """
        Predict rating for user_id on movie_id using Z-Score reconstruction.
        Formula: pred = μ_u + ( (Σ s_ij * z_uj) / Σ|s_ij| ) * σ_u
        
        Returns:
            float: Predicted rating
            NaN: If prediction impossible (no neighbors, no history)
        """
        
        # 1. Check if movie has neighbors
        if movie_id not in self.item_neighbors:
            return np.nan
        
        # 2. Get User Stats
        try:
            user_mean = self.user_means.loc[user_id]
            user_std = self.user_stds.loc[user_id]
        except KeyError:
            # User not in training set
            return np.nan

        # 3. Get Neighbors & User History
        neighbors = self.item_neighbors[movie_id]
        
        # Retrieve user's normalized ratings for these neighbors
        # We access norm_um directly. 
        # Note: neighbors is {movie_id: similarity}
        neighbor_ids = list(neighbors.keys())
        
        # Filter: User must have rated the neighbor (check raw_um for existence)
        # Optimized: check intersection of user_rated and neighbor_ids
        # But efficiently: user_norm slice + check for 0? 
        # Problem: In Z-score, a valid rating can be exactly 0 (if rating == mean).
        # Correct: Check raw_um notna.
        
        user_raw_slice = self.raw_um.loc[user_id]
        valid_neighbors = [m for m in neighbor_ids if pd.notna(user_raw_slice.get(m))]
        
        if not valid_neighbors:
            return np.nan
        
        # 4. Compute Weighted Sum
        # Extract weights (similarities) and values (z-scores)
        weights = np.array([neighbors[m] for m in valid_neighbors])
        z_scores = self.norm_um.loc[user_id, valid_neighbors].values
        
        sum_abs_weights = np.sum(np.abs(weights))
        
        if sum_abs_weights == 0:
            return np.nan
            
        pred_z = np.dot(weights, z_scores) / sum_abs_weights
        
        # 5. Reconstruct Rating
        prediction = user_mean + (pred_z * user_std)
        
        # Clip to valid range (optional, but good for stability 1-5)
        # prediction = np.clip(prediction, 0.5, 5.5) 
        # (User didn't ask for clip, but infinite values check is requested)
        
        if np.isnan(prediction) or np.isinf(prediction):
            return np.nan
            
        return float(prediction)
