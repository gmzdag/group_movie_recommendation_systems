
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

from recommender.UBCF.user_based_cf import UserBasedCF
from recommender.CB.content_based import ContentBasedModel

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



    def recommend_group(self, group_users: List[int], top_n: int = 10, alpha: float = 0.5) -> pd.DataFrame:
        """
        Group Recommendation using Weighted Hybrid Strategy with Mean Aggregation.
        
        Strategy:
        1. For each candidate movie:
           - Calculate Score = (alpha * UBCF_Score) + ((1-alpha) * CBF_Score)
        2. Aggregation: Group Score = Mean(User Scores)
        
        Args:
            alpha (float): Weight for UBCF (0.0 to 1.0). 
                           0.5 means equal weight.
        """
        # Exclude if *any* member watched it to avoid "I've already seen that".
        
        all_watched = set()
        for uid in group_users:
            if uid in self.ubcf.R.index:
                all_watched.update(self.ubcf.R.loc[uid].dropna().index)
        
        all_movies = self.ubcf.R.columns
        candidates = [m for m in all_movies if m not in all_watched]
        
        group_scores = []
        
        # Pre-check neighbors for fallbacks (UBCF might be weak for some)
        # But in Weighted Hybrid, we try to use BOTH.
        
        for mid in candidates:
            member_scores = []
            
            for uid in group_users:
                # --- 1. UBCF Component ---
                ubcf_val = np.nan
                try:
                    # If user has neighbors, predict. Else global mean.
                    if uid in self.ubcf.neighbors and self.ubcf.neighbors[uid]:
                        ubcf_val = self.ubcf.predict(uid, mid)
                    else:
                        ubcf_val = self.ubcf.global_mean 
                except:
                    ubcf_val = self.ubcf.global_mean

                # --- 2. CBF Component (with Year boost) ---
                cbf_val = self.cbf.predict_rating(uid, mid)
                if np.isnan(cbf_val):
                    cbf_val = self.ubcf.global_mean

                # --- 3. Weighted Mix ---
                # Formula: Score = alpha * UBCF + (1-alpha) * CBF
                final_score = (alpha * ubcf_val) + ((1 - alpha) * cbf_val)
                member_scores.append(final_score)
            
            # Aggregation: MEAN STRATEGY
            avg_score = np.mean(member_scores)
            
            group_scores.append({
                "movieId": mid,
                "score": avg_score
            })
            
        # Sort
        group_scores.sort(key=lambda x: x["score"], reverse=True)
        top_items = group_scores[:top_n]
        

        # Add Titles if movie dataframe is available in UBCF or CBF
        # UBCF has self.movies
        results_df = pd.DataFrame(top_items)
        if not results_df.empty and self.ubcf.movies is not None:
             # Ensure index is movieId for mapping
             # Check if movieId is index, if not set it
             if 'movieId' in self.ubcf.movies.columns:
                 title_map = self.ubcf.movies.set_index('movieId')['title']
             else:
                 # Assume it's already index if not column? Safer to rely on what load_movies returns
                 # load_movies returns movieId as a column, default integer index
                 title_map = self.ubcf.movies['title'] # This would be wrong if index != movieId
                 
             # Correct logic:
             # load_movies() returns DF with 'movieId' column.
             title_map = self.ubcf.movies.set_index('movieId')['title']
             results_df['title'] = results_df['movieId'].map(title_map)
             
        return results_df
