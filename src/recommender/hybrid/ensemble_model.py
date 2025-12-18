import numpy as np
import pandas as pd
from typing import List, Dict

class EnsembleRecommender:
    """
    Ensemble Recommender
    --------------------
    Combines predictions from multiple Hybrid Models using a Weighted Score strategy.
    
    Formula:
        FinalScore(item) = Sum(Weight_m * Score_m(item)) for m in Models
    """
    
    def __init__(self, models: List, weights: List[float]=None):
        """
        Args:
            models: List of model instances (must have recommend_for_group method)
            weights: List of floats summing to 1.0 (approx). 
                     If None, initialized to equal weights.
        """
        self.models = models
        if weights:
            self.weights = weights
        else:
            self.weights = [1.0 / len(models)] * len(models)
            
    def set_weights(self, weights: List[float]):
        if len(weights) != len(self.models):
            raise ValueError("Number of weights must match number of models.")
        self.weights = weights
        
    def recommend(self, group_users: List[int], candidates: List[int], top_k: int=10) -> List[Dict]:
        """
        Generates Ensemble Recommendations.
        """
        all_results = {} # mid -> {model_idx: score, 'explanations': ...}
        
        for i, model in enumerate(self.models):
            # Check if model has a 'ubcf' or 'R' attribute to filter watched
            # Or filter globally before calling.
            # Usually filtering is handled by the caller or inside recommend_for_group.
            # But ensure consistency.
            pass
            
        # Global Filtering: Remove if ANY user in group has seen it
        # This requires access to history. The models might not all have it easy.
        # But `HybridModel2` logic had a filter.
        # Let's rely on `recommend` caller passing valid candidates.
        # However, to be safe, if we have access to history, we should filter.
        # Current design: `predict_online.py` generates candidates. We should filter THERE.
        
        for i, model in enumerate(self.models):
             # Pass explicit candidates list
            recs = model.recommend_for_group(group_users, candidates, top_k=len(candidates))
            
            for rec in recs:
                mid = rec['movie_id']
                score = rec['score']
                
                if mid not in all_results:
                    all_results[mid] = {'scores': [0.0]*len(self.models), 'explanations': [None]*len(self.models)}
                
                all_results[mid]['scores'][i] = score
                all_results[mid]['explanations'][i] = rec # Store full rec object for explanation mining
        
        # 2. Aggregation
        final_list = []
        
        for mid, data in all_results.items():
            scores = data['scores']
            
            # Weighted Sum
            final_score = sum(s * w for s, w in zip(scores, self.weights))
            
            # 3. Explain Selection
            # Which model contributed most?
            weighted_scores = [s * w for s, w in zip(scores, self.weights)]
            # argmax
            best_model_idx = np.argmax(weighted_scores)
            best_rec_obj = data['explanations'][best_model_idx]
            
            # Base explanation comes from the dominant model
            # We can enrich it: "Model A (50%) and Model B (30%) both highly rated this."
            if best_rec_obj:
               group_reason = f"[Ensemble] {best_rec_obj.get('group_explanation', '')}"
               individual_expls = best_rec_obj.get('explanations', {})
            else:
               group_reason = "Ensemble Consensus"
               individual_expls = {}

            final_list.append({
                'movie_id': mid,
                'score': final_score,
                'group_explanation': group_reason,
                'explanations': individual_expls,
                'source_model_idx': int(best_model_idx)
            })
            
        # 4. Sort
        final_list.sort(key=lambda x: x['score'], reverse=True)
        return final_list[:top_k]
