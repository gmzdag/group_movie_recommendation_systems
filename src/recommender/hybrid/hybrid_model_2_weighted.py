"""
HYBRID MODEL 2 ALTERNATIVE: Confidence-Weighted Strategy
========================================================

IMPROVEMENT over Binary Switching:
- Smooth transition instead of hard switching
- Uses both models always (no information loss)
- Neighbor count determines confidence/weight

FORMULA (inspired by Hybrid Model 1):
    confidence = n / (n + C)
    score = confidence × UBCF + (1 - confidence) × CBF

Where:
    n = number of UBCF neighbors
    C = trust parameter (controls transition steepness)

REFERENCES:
- Pazzani, M. J. (1999). "A framework for collaborative, content-based and demographic filtering"
- Burke, R. (2002). "Hybrid recommender systems: Survey and experiments"
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any

from recommender.UBCF.user_based_cf import UserBasedCF
from recommender.CB.content_based import ContentBasedModel


class WeightedHybridUBCF_CBF:
    """
    Weighted Hybrid combining UBCF and CBF with confidence-based weighting.
    
    Alternative to binary switching - provides smooth transition.
    """
    
    def __init__(self, ubcf_model: UserBasedCF, cbf_model: ContentBasedModel, C: float = 10.0):
        """
        Args:
            ubcf_model: User-Based Collaborative Filtering model
            cbf_model: Content-Based Filtering model
            C: Trust parameter (default: 10.0)
               - Lower C: Trust UBCF with fewer neighbors  
               - Higher C: Require more neighbors to trust UBCF
               - Literature range: 1-30 (Pazzani, 1999)
        """
        self.ubcf = ubcf_model
        self.cbf = cbf_model
        self.C = C
        
    def predict(self, user_id: int, movie_id: int) -> Tuple[float, str]:
        """
        Predict rating using confidence-weighted combination.
        Returns: (score, method_info)
        """
        # Get neighbor count
        n_neighbors = len(self.ubcf.neighbors.get(user_id, {}))
        
        # Calculate confidence (weight for UBCF)
        confidence = n_neighbors / (n_neighbors + self.C)
        
        # Get both predictions
        try:
            ubcf_pred = self.ubcf.predict(user_id, movie_id)
        except:
            ubcf_pred = np.nan
            
        cbf_pred = self.cbf.predict_rating(user_id, movie_id)
        
        # Handle NaNs
        if np.isnan(ubcf_pred) and np.isnan(cbf_pred):
            return self.ubcf.global_mean, "GlobalMean"
        elif np.isnan(ubcf_pred):
            return cbf_pred, "CBF_only"
        elif np.isnan(cbf_pred):
            return ubcf_pred, "UBCF_only"
        
        # Weighted combination
        score = confidence * ubcf_pred + (1 - confidence) * cbf_pred
        
        method_info = f"Weighted(UBCF:{confidence:.2f}, CBF:{1-confidence:.2f}, n={n_neighbors})"
        
        return score, method_info
    
    def recommend_for_group(self, group_users: List[int], candidates: List[int], 
                           top_k: int = 10) -> List[Dict]:
        """
        Group recommendation using weighted hybrid for each user.
        """
        group_scores = []
        
        for mid in candidates:
            member_scores = []
            
            for uid in group_users:
                try:
                    score, method = self.predict(uid, mid)
                    if not np.isnan(score):
                        member_scores.append(score)
                except:
                    pass
            
            if member_scores:
                avg_score = np.mean(member_scores)
                group_scores.append((mid, avg_score))
        
        # Sort and return top K
        group_scores.sort(key=lambda x: x[1], reverse=True)
        top_items = group_scores[:top_k]
        
        results = []
        for mid, score in top_items:
            results.append({
                'movie_id': mid,
                'score': score,
                'group_explanation': f'Weighted Hybrid (C={self.C})',
                'explanations': {}
            })
        
        return results
