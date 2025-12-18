import numpy as np
import pandas as pd

class HybridModel1:
    """
    Hybrid Model 1: Dynamic Weighted Hybrid
    ---------------------------------------
    Combines Item-Based CF and Content-Based Filtering using a dynamic weight
    based on the number of neighbors available for the Item-Based prediction.
    
    Formula:
        Score = (n / (n + C)) * IB_Score + (C / (n + C)) * CB_Score
        
    Where:
        n = number of item neighbors used (support)
        C = hyperparameter controlling the trust transition (lower C = trust IB sooner)
    """
    
    def __init__(self, ib_model, cb_model, C=1):
        """
        Args:
            ib_model: Instance of ItemBasedCF
            cb_model: Instance of ContentBasedModel
            C (float): Trust factor. Default 1.0 (Optimal based on NDCG validation experiments).
                       - High C: Requires many neighbors to switch to IB.
                       - Low C: Switches to IB with few neighbors.
                       - Empirically, C=1.0 balances Cold Start (CB) and rich history (IB).
        """
        self.ib_model = ib_model
        self.cb_model = cb_model
        self.C = C
        
    def predict(self, user_id, movie_id):
        """
        Predicts rating for a user on a movie using a dynamic weighted hybrid approach.
        """
        # Get Item-Based prediction and neighbor support
        ib_pred, info = self.ib_model.predict(user_id, movie_id, return_info=True)
        neighbors_count = info.get('n_neighbors', 0)
        
        # Get Content-Based prediction (fallback)
        cb_pred = self.cb_model.predict_rating(user_id, movie_id)
        
        # Handle cases where one model fails
        if np.isnan(ib_pred): return cb_pred
        if np.isnan(cb_pred): return ib_pred
            
        # Dynamic Weighting based on Trust (neighbors count)
        n = neighbors_count
        alpha = n / (n + self.C) # Trust IB more as n increases
        beta = self.C / (n + self.C)
        
        return (alpha * ib_pred) + (beta * cb_pred)

    def explain(self, user_id, movie_id):
        """
        Generates explanation signals for the ExplanationEngine.
        """
        from src.recommender.explanation_engine import ExplanationEngine
        
        signals = []
        
        # --- 1. IBCF Signal (Collaborative Behavior only) ---
        try:
            ib_expl = self.ib_model.get_explanation(user_id, movie_id)
            neighbors = ib_expl.get('neighbors', [])
            
            for n in neighbors:
                if n.get('similarity', 0) > 0.1:
                    # STRICT RULE: IBCF is about User Behavior, not content.
                    # We do NOT look up shared traits here.
                    signals.append({
                        'source': 'IBCF',
                        'strength': n.get('similarity', 0.5),
                        'context_items': [n.get('title')],
                        'features': [] # No content features for IBCF
                    })
                    break 
        except Exception:
            pass

        # --- 2. CB Signal (Content Attributes) ---
        try:
            if user_id in self.ib_model.raw_um.index:
                user_hist = self.ib_model.raw_um.loc[user_id]
                liked_items = user_hist[user_hist >= 4.0].index.tolist()
                
                best_match = None
                max_common = 0
                best_traits = []
                
                for hist_mid in liked_items:
                    traits = self.cb_model.get_shared_traits(movie_id, hist_mid)
                    
                    # Collect all meaningful traits (Keywords, Directors, Actors)
                    # Ignore Genres per strict rules unless supported by others? Strict rule said "NOT high-level genres". 
                    # So we rely on Keywords + People.
                    
                    kw = list(traits.get('Keywords', []))
                    directors = list(traits.get('Directors', []))
                    actors = list(traits.get('Actors', []))
                    
                    # Combine meaningful attributes
                    # Prefix people to distinguish in Engine? Or just list them.
                    # "Christopher Nolan" is self-explanatory.
                    
                    current_traits = directors + actors + kw
                    
                    if len(current_traits) > 0:
                        count = len(current_traits)
                        if count > max_common:
                            max_common = count
                            title = self.cb_model.movies_df[self.cb_model.movies_df['movieId'] == hist_mid]['title'].values
                            best_match = title[0] if len(title) > 0 else "a movie you liked"
                            best_traits = current_traits
                
                if max_common > 0 and best_match:
                    signals.append({
                        'source': 'CBF',
                        'strength': 0.6 if max_common >= 2 else 0.4,
                        'context_items': [best_match],
                        'features': best_traits[:3] # Pass top 3 mixed traits
                    })
        except Exception:
            pass
            
        # Use Engine to Format
        return ExplanationEngine.generate_explanation(signals)

    def recommend_for_group(self, user_ids, candidates, top_k=10):
        """
        Generates group recommendations by filtering watched items and averaging scores.
        
        ROBUST: Includes error handling to prevent crashes from individual user failures.
        """
        # Filter out movies watched by any member
        valid_candidates = []
        for mid in candidates:
            watched = False
            try:
                for uid in user_ids:
                    if uid in self.ib_model.raw_um.index and mid in self.ib_model.raw_um.columns:
                        if pd.notna(self.ib_model.raw_um.loc[uid, mid]):
                            watched = True
                            break
            except Exception as e:
                print(f"[WARNING] Error checking watched status for movie {mid}: {e}")
                continue
            
            if not watched:
                valid_candidates.append(mid)
        
        # Filter sequels to prevent spoilers (unless prequel is seen)
        try:
            valid_candidates = self._filter_sequels(valid_candidates, user_ids)
        except Exception as e:
            print(f"[WARNING] Sequel filtering failed: {e}. Continuing without sequel filter.")
                
        # Calculate Average Group Score
        group_results = []
        for mid in valid_candidates:
            scores = []
            for uid in user_ids:
                try:
                    s = self.predict(uid, mid)
                    if not np.isnan(s):
                        scores.append(s)
                except Exception as e:
                    print(f"[WARNING] Prediction failed for user {uid}, movie {mid}: {e}")
                    continue
            
            if scores:
                group_results.append((mid, np.mean(scores)))
                
        group_results.sort(key=lambda x: x[1], reverse=True)
        top_items = group_results[:top_k]
        
        # Generate Explanations for Top Items
        final_recommendations = []
        for mid, score in top_items:
            explanations = {}
            for uid in user_ids:
                try:
                    explanations[uid] = self.explain(uid, mid)
                except Exception as e:
                    print(f"[WARNING] Explanation failed for user {uid}, movie {mid}: {e}")
                    explanations[uid] = {
                        'primary_reason': 'Recommended based on overall similarity.',
                        'secondary_reasons': [],
                        'confidence_level': 'Weak',
                        'signal_source': 'Unknown'
                    }
            
            try:
                group_reason = self._generate_group_explanation(explanations)
            except Exception as e:
                print(f"[WARNING] Group explanation failed for movie {mid}: {e}")
                group_reason = "Recommended for the group."
            
            final_recommendations.append({
                'movie_id': mid,
                'score': score,
                'group_explanation': group_reason,
                'explanations': explanations
            })
            
        return final_recommendations

    def _generate_group_explanation(self, explanations):
        """
        Synthesizes a group-level explanation based on individual structured explanations.
        Looks for common patterns across 'features' lists in secondary reasons or signals.
        """
        from collections import Counter
        
        # 1. Analyze Sources
        sources = [e.get('signal_source', 'Unknown') for e in explanations.values()]
        # Handle empty explanations
        if not sources: return "Recommended based on group popularity."
        
        common_source = Counter(sources).most_common(1)[0][0]
        
        # 2. Extract and Count Features (Traits/People) from the Explanation text or we need raw signals?
        # The 'explanations' dict passed here is the OUTPUT of explain(), which is JSON.
        # Structure: {'primary_reason': ..., 'secondary_reasons': [...], 'signal_source': ...}
        # It DOES NOT return the raw 'features' list directly in the top level, 
        # but ExplanationEngine formatted them into text.
        # To do this accurately, we should probably have returned 'evidence' or 'features' in the JSON from explain().
        # Let's adjust explain() return? 
        # Actually HybridModel1.explain() used ExplanationEngine.generate_explanation().
        # That method returns { ..., "evidence": ...? } No, check implementation.
        # It returns signal_source, confidence_level, primary/secondary text.
        # We need to extract commonalities.
        # Let's assume we can regex the text or we update ExplanationEngine to pass 'features' through.
        # Updating ExplanationEngine is cleaner, but let's try to parse "focus on X, Y" from text 
        # OR just assume we can access more data.
        
        # Wait, the `explanations` input to this method comes from `recommend_for_group` loop:
        # `explanations = {uid: self.explain(uid, mid) for uid in user_ids}`
        # `self.explain` returns the Engine output.
        
        # Let's try to infer from the 'secondary_reasons' text which often contains the features for CBF.
        # "Recommended for its focus on Nolan, time travel."
        # Regex: "focus on (.*)" or "shared themes like (.*)"
        
        import re
        all_traits = []
        for e in explanations.values():
            for reason in [e.get('primary_reason', '')] + e.get('secondary_reasons', []):
                # Try to catch list of traits
                # CBF Pattern 1: "...focus on (.*)."
                m1 = re.search(r"focus on (.*?)(\.|,|$)", reason)
                if m1: 
                    traits = m1.group(1).split(",")
                    all_traits.extend([t.strip().title() for t in traits])
                    
                # CBF Pattern 2: "...themes like (.*)."
                m2 = re.search(r"themes like (.*?)(\.|,|$)", reason)
                if m2:
                    traits = m2.group(1).split(",")
                    all_traits.extend([t.strip().title() for t in traits])
                    
        # 3. Find Consensus
        common_trait = None
        if all_traits:
            counts = Counter(all_traits)
            most_common = counts.most_common(1)[0]
            if most_common[1] >= max(2, len(explanations) // 2): # At least 2 or half group
                common_trait = most_common[0]
        
        # 4. Construct Narrative
        if common_source == 'IBCF':
            return "Recommended because multiple members have watched similar movies."
        elif common_source == 'CBF':
            if common_trait:
                return f"Aligns with the group's interest in {common_trait}."
            return "Aligns with the content preferences (themes/cast) of the group."
        elif common_source == 'WATCHLIST':
            return "Matches one or more user watchlists."
        
        return "Broadly appealing to the group's diverse tastes."

    def _filter_sequels(self, candidates, user_ids):
        """
        Removes sequel candidates if the prequel has not been watched by the group.
        Uses title stemming to identify series.
        """
        movies = self.cb_model.movies_df
        
        # Identify all watched movies by the group
        watched_ids = set()
        for uid in user_ids:
            if uid in self.ib_model.raw_um.index:
                user_hist = self.ib_model.raw_um.loc[uid]
                watched_ids.update(user_hist[user_hist.notna()].index.tolist())
                
        cand_meta = movies[movies['movieId'].isin(candidates)].copy()
        
        def get_stem(title):
            import re
            t = title.lower().strip()
            if ":" in t: return t.split(":")[0].strip()
            t = re.sub(r'\s+\d+$', '', t) # Remove trailing numbers
            t = re.sub(r'\s+(ii|iii|iv|v|vi|vii|viii|ix|x)$', '', t) # Remove Roman numerals
            return t.strip()
            
        cand_meta['stem'] = cand_meta['title'].apply(get_stem)
        stems = set(cand_meta['stem'].unique())
        
        # Check all movies in DB for these stems to find prequels
        all_movies = movies.copy()
        all_movies['stem'] = all_movies['title'].apply(get_stem)
        related = all_movies[all_movies['stem'].isin(stems)].sort_values(['stem', 'year'])
        
        invalid_candidates = set()
        for stem, group in related.groupby('stem'):
            if len(group) < 2: continue
            
            sorted_mids = group['movieId'].values
            for i in range(1, len(sorted_mids)):
                current_mid = sorted_mids[i]
                prev_mid = sorted_mids[i-1]
                
                # If we recommend a sequel, ensure prequel is watched
                if current_mid in candidates and prev_mid not in watched_ids:
                    invalid_candidates.add(current_mid)
                        
        return [c for c in candidates if c not in invalid_candidates]
