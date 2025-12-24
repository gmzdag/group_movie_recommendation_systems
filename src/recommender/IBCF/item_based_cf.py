"""
Item-Based Collaborative Filtering (Signal Provider)
----------------------------------------------------
Optimized Configuration (Validated via Grid Search):
- Normalization: Z-SCORE
- Similarity: COSINE (Adjusted Cosine)
- Min Ratings Filter: 10 (items with <10 ratings excluded)
- Neighborhood Size (K): 60 neighbors
- Prediction: mean + (weighted_sum * std)
- Performance: NDCG@10 = 0.2123 (Validation), 0.2069 (Test)
- Output: Single prediction score (or NaN)
"""

import numpy as np
import pandas as pd


class ItemBasedCF:
    def __init__(self, raw_um, norm_um, item_neighbors, movies, top_k=60):
        """
        Args:
            raw_um: Raw Rating Matrix (Users x Movies), Missing=NaN
            norm_um: Z-Score Normalized Matrix, Missing=0
            item_neighbors: Pre-computed top-K neighbors dict
            movies: Movies metadata (reference)
            top_k: Neighbor count (default=60, validated optimal) 
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
    def predict(self, user_id, movie_id, verbose=False, return_info=False):
        """
        Predict rating for user_id on movie_id using Z-Score reconstruction.
        Formula: pred = μ_u + ( (Σ s_ij * z_uj) / Σ|s_ij| ) * σ_u
        
        Args:
            return_info (bool): If True, returns tuple (prediction, info_dict)
        
        Returns:
            float: Predicted rating
            NaN: If prediction impossible (no neighbors, no history)
        """
        
        info = {'n_neighbors': 0}

        # 1. Check if movie has neighbors
        if movie_id not in self.item_neighbors:
            return (np.nan, info) if return_info else np.nan
        
        # 2. Get User Stats
        try:
            user_mean = self.user_means.loc[user_id]
            user_std = self.user_stds.loc[user_id]
        except KeyError:
            # User not in training set
            return (np.nan, info) if return_info else np.nan
        
        # 3. Get Neighbors & User History
        neighbors = self.item_neighbors[movie_id]
        neighbor_ids = list(neighbors.keys())
        
        user_raw_slice = self.raw_um.loc[user_id]
        valid_neighbors = [m for m in neighbor_ids if pd.notna(user_raw_slice.get(m))]
        
        info['n_neighbors'] = len(valid_neighbors)
        
        if not valid_neighbors:
            return (np.nan, info) if return_info else np.nan
        
        # 4. Compute Weighted Sum
        weights = np.array([neighbors[m] for m in valid_neighbors])
        z_scores = self.norm_um.loc[user_id, valid_neighbors].values
        
        sum_abs_weights = np.sum(np.abs(weights))
        
        if sum_abs_weights == 0:
            return (np.nan, info) if return_info else np.nan
            
        pred_z = np.dot(weights, z_scores) / sum_abs_weights
        
        # 5. Reconstruct Rating
        prediction = user_mean + (pred_z * user_std)
        
        if np.isnan(prediction) or np.isinf(prediction):
            return (np.nan, info) if return_info else np.nan
            
        result = float(prediction)
        return (result, info) if return_info else result

    def get_explanation(self, user_id, movie_id, top_k=5):
        """
        Explains why a movie was recommended based on Item-Based CF.
        Return top neighbors that the user reacted to.
        """
        explanation = {'type': 'IB', 'neighbors': []}
        
        # 1. Check if movie has neighbors
        if movie_id not in self.item_neighbors:
            return explanation
            
        # 2. Get User History to check what they rated
        user_raw_slice = self.raw_um.loc[user_id]
        
        # 3. Get Neighbors
        neighbors = self.item_neighbors[movie_id] # {mid: sim}
        
        # Filter: User must have rated
        valid_neighbors = []
        for mid, sim in neighbors.items():
            if pd.notna(user_raw_slice.get(mid)):
                rating = user_raw_slice.get(mid)
                valid_neighbors.append({
                    'id': mid,
                    'sim': sim,
                    'rating': rating,
                    'contribution': sim * rating # Approximation of contribution
                })
        
        # Sort by contribution (Similarity * Rating)
        valid_neighbors.sort(key=lambda x: x['contribution'], reverse=True)
        
        # Return Top K
        top_n = valid_neighbors[:top_k]
        
        # Resolve titles? We have self.movies
        for n in top_n:
            try:
                title = self.movies.loc[self.movies['movieId'] == n['id'], 'title'].values[0]
            except:
                title = f"Movie {n['id']}"
                
            explanation['neighbors'].append({
                'id': n['id'],
                'title': title,
                'rating': n['rating'],
                'similarity': n['sim']
            })
            
        return explanation
