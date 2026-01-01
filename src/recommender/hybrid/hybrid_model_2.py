
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any

from src.recommender.UBCF.user_based_cf import UserBasedCF
from src.recommender.CB.content_based import ContentBasedModel


class SwitchingHybridRecommender:
    """
    Hybrid Model 2: UBCF + CBF with Optimized Performance-Weighted Strategy
    
    DEFAULT STRATEGY: Performance-Weighted (w_ubcf=0.10, w_cbf=0.90)
    - Test NDCG@10: 0.288 (Group Recommendations)
    - Optimized via group-level validation
    
    ALTERNATIVE: Binary Switching (for comparison/ablation studies)
    
    References:
    - experiments/hybrid2_group_optimization.py
    - experiments/hybrid_models_test_evaluation.py
    """
    
    def __init__(self, ubcf_model: UserBasedCF, cbf_model: ContentBasedModel, 
                 strategy: str = 'performance_weighted',
                 neighbor_threshold: int = 30,
                 w_ubcf: float = 0.10,
                 w_cbf: float = 0.90):
        """
        Args:
            ubcf_model: User-Based Collaborative Filtering model
            cbf_model: Content-Based Filtering model  
            strategy: 'performance_weighted' (default, optimal) or 'switching'
            neighbor_threshold: For switching - min neighbors to use UBCF (default: 30)
            w_ubcf: UBCF weight for performance_weighted (default: 0.05)
            w_cbf: CBF weight for performance_weighted (default: 0.95)
        """
        self.ubcf = ubcf_model
        self.cbf = cbf_model
        self.strategy = strategy
        self.neighbor_threshold = neighbor_threshold
        self.w_ubcf = w_ubcf
        self.w_cbf = w_cbf
        
    def predict(self, user_id: int, movie_id: int) -> Tuple[float, str]:
        """
        Predict rating using configured strategy.
        Returns: (score, source_method)
        """
        if self.strategy == 'performance_weighted':
            # OPTIMAL: Weighted combination
            try:
                ubcf_pred = self.ubcf.predict(user_id, movie_id)
            except:
                ubcf_pred = np.nan
            
            cbf_pred = self.cbf.predict_rating(user_id, movie_id)
            
            if np.isnan(ubcf_pred) and np.isnan(cbf_pred):
                return self.ubcf.global_mean, "GlobalMean"
            elif np.isnan(ubcf_pred):
                return cbf_pred, "CBF_only"
            elif np.isnan(cbf_pred):
                return ubcf_pred, "UBCF_only"
            
            # Weighted combination
            score = self.w_ubcf * ubcf_pred + self.w_cbf * cbf_pred
            return score, f"Weighted({self.w_ubcf:.2f}/{self.w_cbf:.2f})"
        
        else:  # Binary switching
            try:
                if user_id not in self.ubcf.neighbors or len(self.ubcf.neighbors.get(user_id, {})) < self.neighbor_threshold:
                    raise ValueError(f"Insufficient neighbors")
                    
                prediction = self.ubcf.predict(user_id, movie_id)
                return prediction, "UBCF"
                
            except (KeyError, ValueError):
                cbf_score = self.cbf.predict_rating(user_id, movie_id)
                
                if np.isnan(cbf_score):
                    return self.ubcf.global_mean, "GlobalMean"
                    
                return cbf_score, "CBF"

    def recommend(self, user_id: int, top_n: int = 10) -> pd.DataFrame:
        """Generate recommendations for single user."""
        all_movies = self.ubcf.R.columns
        
        if user_id in self.ubcf.R.index:
            rated_movies = self.ubcf.R.loc[user_id].dropna().index
        else:
            rated_movies = []
            
        candidates = [m for m in all_movies if m not in rated_movies]
        
        results = []
        for mid in candidates:
            score, method = self.predict(user_id, mid)
            results.append({
                "movieId": mid,
                "score": score,
                "method": method
            })
            
        results.sort(key=lambda x: x["score"], reverse=True)
        return pd.DataFrame(results[:top_n])

    def recommend_for_group(self, group_users: List[int], candidates: List[int], 
                           top_k: int = 10, alpha: float = 0.5) -> List[Dict]:
        """
        Group recommendation using configured strategy.
        Returns list of dicts for ensemble compatibility.
        """
        # 1. Pre-compute UBCF scores (Batch per user)
        ubcf_batch = {}
        if hasattr(self.ubcf, 'predict_for_user'):
            for uid in group_users:
                # Optimized vector prediction
                ubcf_batch[uid] = self.ubcf.predict_for_user(uid, candidates)
        else:
            # Fallback (shouldn't happen with correct class)
            ubcf_batch = {uid: {} for uid in group_users}
            
        # 2. Pre-compute CBF scores (Batch group)
        try:
            cbf_batch = self.cbf.predict_for_group(group_users, candidates)
        except AttributeError:
            cbf_batch = {}

        group_scores = []
        
        for mid in candidates:
            member_scores = []
            
            for uid in group_users:
                # Retrieve pre-calculated
                ubcf_score = ubcf_batch.get(uid, {}).get(mid, np.nan)
                cbf_score = cbf_batch.get(mid, {}).get(uid, np.nan)
                
                final_score = np.nan
                
                # Logic from self.predict()
                if self.strategy == 'performance_weighted':
                    if np.isnan(ubcf_score) and np.isnan(cbf_score):
                        final_score = self.ubcf.global_mean
                    elif np.isnan(ubcf_score):
                        final_score = cbf_score
                    elif np.isnan(cbf_score):
                        final_score = ubcf_score
                    else:
                        final_score = self.w_ubcf * ubcf_score + self.w_cbf * cbf_score
                        
                else: # Switching
                    # Check neighbor threshold
                    user_neighbors = self.ubcf.neighbors.get(uid, {})
                    if len(user_neighbors) >= self.neighbor_threshold:
                         # Use UBCF
                         final_score = ubcf_score
                         if np.isnan(final_score): # If UBCF failed despite neighbors
                             final_score = cbf_score
                    else:
                         # Use CBF
                         final_score = cbf_score
                         
                    if np.isnan(final_score):
                         final_score = self.ubcf.global_mean

                if not np.isnan(final_score):
                    member_scores.append(final_score)
            
            if member_scores:
                avg_score = np.mean(member_scores)
                group_scores.append((mid, avg_score))
        
        group_scores.sort(key=lambda x: x[1], reverse=True)
        top_items = group_scores[:top_k]
        
        results = []
        for mid, score in top_items:
            # Generate explanations
            item_explanations = {}
            for uid in group_users:
                try:
                    item_explanations[uid] = self.explain(uid, mid)
                except Exception:
                    item_explanations[uid] = "Recommended based on group stats."

            # Determine signal source based on strategy
            sig_source = "Hybrid (Weighted)" if self.strategy == 'performance_weighted' else "Hybrid (Switching)"
            
            results.append({
                'movie_id': mid,
                'score': score,
                'group_explanation': 'Aligns with the common tastes and shared movie preferences of the group.',
                'explanations': item_explanations,
                'signal_source': sig_source
            })
        
        return results
    
    def recommend_group(self, group_users: List[int], top_n: int = 10, 
                       candidates: List[int] = None) -> pd.DataFrame:
        """
        Legacy group recommendation (returns DataFrame).
        """
        if candidates is None:
            all_watched = set()
            for uid in group_users:
                if uid in self.ubcf.R.index:
                    all_watched.update(self.ubcf.R.loc[uid].dropna().index)
            
            all_movies = self.ubcf.R.columns
            candidates = [m for m in all_movies if m not in all_watched]
        
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
                group_scores.append({"movieId": mid, "score": avg_score})
        
        group_scores.sort(key=lambda x: x["score"], reverse=True)
        
        return pd.DataFrame(group_scores[:top_n])

    def explain(self, user_id: int, movie_id: int) -> Dict[str, Any]:
        """Generate explanation for prediction."""
        from recommender.explanation_engine import ExplanationEngine
        
        signals = []
        score, method = self.predict(user_id, movie_id)
        
        if 'UBCF' in method:
            n_count = len(self.ubcf.neighbors.get(user_id, {}))
            strength = min(1.0, n_count / 50.0)
            signals.append({
                'source': 'UBCF',
                'strength': strength,
                'context_items': [],
                'features': []
            })
        
        if 'CBF' in method or self.strategy == 'performance_weighted':
            signals.append({
                'source': 'CBF',
                'strength': self.w_cbf if self.strategy == 'performance_weighted' else 1.0,
                'context_items': [],
                'features': []
            })
        
        return ExplanationEngine.generate_explanation(signals)
