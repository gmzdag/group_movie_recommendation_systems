
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any

from src.recommender.UBCF.user_based_cf import UserBasedCF
from src.recommender.CB.content_based import ContentBasedModel


class SwitchingHybridRecommender:
    """
    Switching Hybrid Recommender System.
    
    Strategy:
    1. Primary: User-Based Collaborative Filtering (UBCF)
       - Used when the user has enough history and neighbors.
    
    2. Fallback (Switch): Content-Based Filtering (CBF)
       - Used when UBCF fails (Cold Start, no neighbors, or prediction impossible).
       
    This handles the Cold Start problem effectively by leveraging item metadata
    when user interaction data is sparse.
    """
    
    def __init__(self, ubcf_model: UserBasedCF, cbf_model: ContentBasedModel):
        self.ubcf = ubcf_model
        self.cbf = cbf_model
        
    def predict(self, user_id: int, movie_id: int) -> Tuple[float, str]:
        """
        Predict rating using Switching strategy.
        Returns: (score, source_method)
        """
        # Try UBCF first
        try:
            # We assume UBCF returns a valid score or global mean
            # To detect if it 'failed' (e.g. returned global mean due to no neighbors),
            # we might need to check internal state or assume global_mean implies weakness.
            # But strictly speaking, UBCF methods usually handle this gracefully.
            # Let's verify simply: if user has no neighbors, switch.
            
            # Check for Cold Start / No Neighbors in UBCF
            if user_id not in self.ubcf.neighbors or not self.ubcf.neighbors[user_id]:
                raise ValueError("No neighbors (Cold Start)")
                
            prediction = self.ubcf.predict(user_id, movie_id)
            
            # If UBCF fell back to global mean but we wanted more personalization, 
            # we could verify if prediction == global_mean. But let's trust UBCF logic for now.
            return prediction, "UBCF"
            
        except (KeyError, ValueError):
            # Switch to Content-Based
            # CBF uses user profile constructed from their limited ratings (if any)
            # If user has absolutely 0 ratings, CBF might return global mean too.
            cbf_score = self.cbf.predict_rating(user_id, movie_id)
            
            if np.isnan(cbf_score):
                return self.ubcf.global_mean, "GlobalMean" # Both failed
                
            return cbf_score, "CBF"

    def recommend(self, user_id: int, top_n: int = 10) -> pd.DataFrame:
        """
        Generate recommendations for a single user using the best available method.
        """
        # 1. Candidate Generation
        # Get all movies
        all_movies = self.ubcf.R.columns
        
        # Exclude watched
        if user_id in self.ubcf.R.index:
            rated_movies = self.ubcf.R.loc[user_id].dropna().index
        else:
            rated_movies = []
            
        candidates = [m for m in all_movies if m not in rated_movies]
        
        results = []
        
        # 2. Prediction Loop
        # We determine the method ONCE for the user to stay consistent, 
        # or per item? Switching usually implies User-Level switching.
        
        method = "UBCF"
        if user_id not in self.ubcf.neighbors or not self.ubcf.neighbors[user_id]:
            method = "CBF"
        
        # If we selected CBF because of cold start, check if we can even build a profile
        if method == "CBF":
            # Check if user has ANY ratings for CBF
            user_hist = self.cbf.ratings_df[self.cbf.ratings_df['userId'] == user_id]
            if user_hist.empty:
                method = "POPULARITY" # Complete Cold Start
        
        for mid in candidates:
            score = 0.0
            used_method = method
            
            if method == "UBCF":
                score = self.ubcf.predict(user_id, mid)
            elif method == "CBF":
                score = self.cbf.predict_rating(user_id, mid)
                if np.isnan(score):
                    score = self.ubcf.global_mean
            else:
                # Popularity fallback (using item means from UBCF)
                score = self.ubcf.item_means.get(mid, self.ubcf.global_mean)
            
            results.append({
                "movieId": mid,
                "score": score,
                "method": used_method
            })
            
        # 3. Sort and Return
        results.sort(key=lambda x: x["score"], reverse=True)
        top_results = results[:top_n]
        
        return pd.DataFrame(top_results)



    def recommend_for_group(self, group_users: List[int], candidates: List[int], top_k: int = 10, alpha: float = 0.5) -> List[Dict]:
        """
        Alias for recommend_group to match HybridModel1 interface.
        Returns list of dicts instead of DF.
        """
        # Call internal method (adapted)
        # SwitchingHybrid recommend_group signature: (group_users, top_n, alpha)
        # It returns DataFrame. We need List[Dict] to match Ensemble expectation.
        
        # Adaptation:
        df = self.recommend_group(group_users, top_n=top_k, candidates=candidates)
        
        # Convert to list of dicts
        results = []
        for idx, row in df.iterrows():
            results.append({
                'movie_id': row['movieId'],
                'score': row['score'],
                'group_explanation': "Switching Hybrid Result", # Hybrid 2 doesn't retain explanation detail
                'explanations': {} 
            })
        return results

    def explain(self, user_id: int, movie_id: int) -> Dict[str, Any]:
        """
        Generates explanation signals for Switching Hybrid.
        Checks which method (UBCF/CBF) would be active for this user/item.
        """
        from src.recommender.explanation_engine import ExplanationEngine
        
        signals = []
        
        # Determine likely method (logic mirrors predict loop)
        method = "UBCF"
        if user_id not in self.ubcf.neighbors or not self.ubcf.neighbors[user_id]:
            method = "CBF"
            
        if method == "UBCF":
            # UBCF Signal
            # We don't have detailed "Similar User" names usually (privacy/system design),
            # but we can say "Popular among similar users".
            # Strength? Use prediction vs global mean gap?
            # Or just neighbor count confidence.
            
            n_count = len(self.ubcf.neighbors.get(user_id, {}))
            strength = 0.8 if n_count > 10 else 0.5
            
            signals.append({
                'source': 'UBCF',
                'strength': strength,
                'context_items': [],
                'features': []
            })
            
        else:
            # CBF Signal
            # Fallback to CB logic (similar to Hybrid 1 but maybe simpler here)
            # Re-use CB model to find match
             try:
                # Reuse the logic from Hybrid 1 via CB model helper if available?
                # Or just basic checks.
                # Let's assume CBF model has helper `get_shared_traits`
                pass
             except:
                pass
             
             signals.append({
                'source': 'CBF',
                'strength': 0.4, # Fallback is usually weaker confidence
                'context_items': [],
                'features': []
             })
             
        return ExplanationEngine.generate_explanation(signals)
    
    def recommend_group(self, group_users: List[int], top_n: int = 10, candidates: List[int] = None) -> pd.DataFrame:
        """
        Group Recommendation using True Switching Strategy.
        For each user, we determine the best model (UBCF or CBF) based on data availability,
        then predict the score. Finally, we aggregate these scores for the group.
        """
        # Exclude if *any* member watched it
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
                score = 0.0
                method_used = "UBCF"
                
                # Check for Neighbors (Switching Condition)
                # If user has sufficient neighbors, we trust UBCF (Collaborative).
                # Otherwise, we switch to CBF (Content-Based) to handle Cold Start / Sparsity.
                has_neighbors = (uid in self.ubcf.neighbors and len(self.ubcf.neighbors[uid]) > 0)
                
                if has_neighbors:
                    try:
                        pred = self.ubcf.predict(uid, mid)
                        score = pred
                    except:
                        # Fallback to CBF if UBCF fails technically
                        score = self.cbf.predict_rating(uid, mid)
                        method_used = "CBF"
                else:
                    # Cold Start: Switch to Content-Based
                    score = self.cbf.predict_rating(uid, mid)
                    method_used = "CBF"
                
                if np.isnan(score):
                    score = self.ubcf.global_mean
                    method_used = "GlobalMean"
                    
                member_scores.append(score)
            
            # Aggregation: MEAN STRATEGY
            avg_score = np.mean(member_scores)
            
            group_scores.append({
                "movieId": mid,
                "score": avg_score
            })
            
        # Sort
        group_scores.sort(key=lambda x: x["score"], reverse=True)
        top_items = group_scores[:top_n]
        
        results_df = pd.DataFrame(top_items)
        if not results_df.empty and self.ubcf.movies is not None:
             if 'movieId' in self.ubcf.movies.columns:
                 title_map = self.ubcf.movies.set_index('movieId')['title']
             else:
                 title_map = self.ubcf.movies['title']
                 
             title_map = self.ubcf.movies.set_index('movieId')['title']
             results_df['title'] = results_df['movieId'].map(title_map)
             
        return results_df
