import numpy as np
import pandas as pd
from typing import List, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity
from ..CB.content_based import ContentBasedModel

class WatchlistHybridModel:
    """
    Hybrid Model 3: Watchlist-Driven Content Filtering
    --------------------------------------------------
    Focuses on "Future Intent" by recommending movies similar to what users 
    have explicitly added to their Watchlists, rather than what they have rated.
    
    Score Scale: Rescaled to 0-5 to match other models (Cosine Sim * 5).
    """
    
    def __init__(self, movies_df: pd.DataFrame, watchlist_df: pd.DataFrame, cb_model: ContentBasedModel):
        self.movies_df = movies_df
        self.watchlist_df = watchlist_df
        self.cb_model = cb_model
        
        # Precompute title map
        self.title_map = self.movies_df.set_index('movieId')['title'].to_dict()
        
    def predict(self, user_id: int, movie_id: int) -> float:
        """
        Predicts score based on similarity to User's Watchlist.
        Returns: 0.0 to 5.0
        """
        # 1. Get User Watchlist
        user_wl = self.watchlist_df[self.watchlist_df['userId'] == user_id]['movieId'].unique()
        if len(user_wl) == 0:
            return np.nan # Cold Start
            
        if movie_id not in self.cb_model.movie_to_idx:
            return np.nan
            
        indices = [self.cb_model.movie_to_idx[m] for m in user_wl if m in self.cb_model.movie_to_idx]
        if not indices:
            return np.nan

        if movie_id not in self.cb_model.movie_to_idx:
            return np.nan
            
        indices = [self.cb_model.movie_to_idx[m] for m in user_wl if m in self.cb_model.movie_to_idx]
        if not indices:
            return np.nan

        # Calculate Max Similarity (Best Match Strategy)
        # Instead of averaging the user profile, we find the closest movie in the watchlist.
        watchlist_vectors = self.cb_model.tfidf_matrix[indices] # Sparse (K, F)
        
        idx = self.cb_model.movie_to_idx[movie_id]
        movie_vec = self.cb_model.tfidf_matrix[idx] # Sparse (1, F)
        
        # Compute cosine similarity between target movie and ALL watchlist movies
        # returns array of shape (K, 1)
        sims = cosine_similarity(watchlist_vectors, movie_vec).flatten()
        
        if len(sims) == 0:
            return 0.0
            
        max_sim = sims.max() # Best single match
        
        # Scale to 5
        return max_sim * 5.0

    def explain(self, user_id, movie_id):
        """
        Generates Watchlist explanation signals.
        """
        from ..explanation_engine import ExplanationEngine
        
        signals = []
        
        # Check Watchlist match
        user_wl = self.watchlist_df[self.watchlist_df['userId'] == user_id]['movieId'].tolist()
        
        # Is this specific movie in watchlist?
        if movie_id in user_wl:
             signals.append({
                'source': 'WATCHLIST',
                'strength': 1.0, 
                'context_items': [],
                'features': ["Direct Match"]
             })
        else:
            # Did we find it via similarity to a watchlist item?
            # We need to scan watchlist again to find the 'source' (slow, but needed for explain)
            best_sim = 0
            best_source_id = None
            
            # Optimization: If we stored this during predict, it would be faster.
            # Rerun similarity search for explanation context
            try:
                mov_vec = self.cb_model.tfidf_matrix[self.cb_model.movie_to_idx[movie_id]]
                
                for wl_id in user_wl:
                    if wl_id in self.cb_model.movie_to_idx:
                        wl_vec = self.cb_model.tfidf_matrix[self.cb_model.movie_to_idx[wl_id]]
                        sim = (mov_vec @ wl_vec.T).toarray()[0][0]
                        if sim > best_sim:
                            best_sim = sim
                            best_source_id = wl_id
            except:
                pass
                
            if best_source_id:
                # Get Title
                title = self.cb_model.movies_df[self.cb_model.movies_df['movieId'] == best_source_id]['title'].values
                title_str = title[0] if len(title) > 0 else "Watchlist Item"
                
                signals.append({
                    'source': 'WATCHLIST',
                    'strength': best_sim, 
                    'context_items': [title_str],
                    'features': []
                })

        return ExplanationEngine.generate_explanation(signals)    

    def predict_for_group(self, user_ids, candidates):
        """
        Optimized batch prediction for group.
        Returns: {mid: {uid: score}}
        """
        results = {} 
        
        # 0. Prep candidates
        valid_cands = [m for m in candidates if m in self.cb_model.movie_to_idx]
        if not valid_cands: return {}
        
        cand_indices = [self.cb_model.movie_to_idx[m] for m in valid_cands]
        cand_vecs = self.cb_model.tfidf_matrix[cand_indices] # (N_c, F)
        
        for uid in user_ids:
            # 1. Get Watchlist Vectors
            user_wl = self.watchlist_df[self.watchlist_df['userId'] == uid]['movieId'].unique()
            valid_wl = [m for m in user_wl if m in self.cb_model.movie_to_idx]
            
            if not valid_wl: continue
            
            wl_indices = [self.cb_model.movie_to_idx[m] for m in valid_wl]
            wl_vecs = self.cb_model.tfidf_matrix[wl_indices] # (N_w, F)
            
            # 2. Compute Similarities
            # (N_c, F) @ (N_w, F).T -> (N_c, N_w)
            sim_matrix = cosine_similarity(cand_vecs, wl_vecs)
            
            # 3. Max Sim per candidate
            # np.max(sim_matrix, axis=1) -> (N_c,)
            best_sims = sim_matrix.max(axis=1) # Dense array
            
            # 4. Store
            scores = best_sims * 5.0
            
            for i, mid in enumerate(valid_cands):
                if mid not in results: results[mid] = {}
                results[mid][uid] = float(scores[i])
                
        return results

    def recommend_for_group(self, user_ids: List[int], candidates: List[int], top_k: int=10) -> List[Dict]:
        """
        Group Recommendation Logic: Average of Individual Watchlist Scores.
        """
        # Batch Predict
        batch_scores = self.predict_for_group(user_ids, candidates)

        scores_list = []
        
        for mid in candidates:
            # Check if we have scores
            if mid not in batch_scores:
                continue
                
            user_scores_map = batch_scores[mid]
            if not user_scores_map:
                continue
            
            # Aggregate valid scores
            # user_scores_map only contains valid prediction > 0 or at least calculated
            
            # We must be careful: if a user has NO watchlist items, they are skipped.
            # Avg should be over users who HAVE data? Or penalize?
            # Existing code: `if not np.isnan(s): user_scores.append(s)`
            # My Batch logic skipped users with no watchlist interactions? No, it skipped users with no valid watchlist items.
            # So if user in user_scores_map, they have a score.
            
            vals = list(user_scores_map.values())
            if vals:
                avg_score = np.mean(vals)
                scores_list.append((mid, avg_score))
                
        # Sort
        scores_list.sort(key=lambda x: x[1], reverse=True)
        top_items = scores_list[:top_k]
        
        results = []
        for mid, score in top_items:
            # Generate Explanations
            explanations = {uid: self.explain(uid, mid) for uid in user_ids}
            
            # Group Narrative
            # Check if it was in ANY watchlist specifically (Perfect Match)
            in_any_wl = False
            wl_owners = []
            for uid in user_ids:
                user_wl = self.watchlist_df[self.watchlist_df['userId'] == uid]['movieId'].values
                if mid in user_wl:
                    in_any_wl = True
                    wl_owners.append(str(uid))
            
            if in_any_wl:
                group_reason = f"Explicitly requested by member(s) {', '.join(wl_owners)}."
            else:
                group_reason = "Matches the collective future interests of the group based on watchlists."
            
            results.append({
                'movie_id': mid,
                'score': score, 
                'group_explanation': group_reason,
                'explanations': explanations
            })
            
        return results
