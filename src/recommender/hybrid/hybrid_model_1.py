import numpy as np
import pandas as pd

class HybridModel1:
    """
    Hybrid Model 1: Item-Based CF + Content-Based (Weighted Hybrid)
    ----------------------------------------------------------------
    
    **HYBRIDIZATION MECHANISM**: Dynamic Weighted Hybrid
    - Combines Item-Based Collaborative Filtering (IBCF) with Content-Based Filtering (CBF)
    - Uses neighbor support (n) to dynamically weight between IBCF and CBF
    
    **INDIVIDUAL PREDICTION FORMULA**:
        α(n) = n / (n + C)      # IBCF weight
        β(n) = C / (n + C)      # CBF weight
        score(u, i) = α(n) * IBCF(u, i) + β(n) * CBF(u, i)
        
    Where:
        n = number of item neighbors used in IBCF prediction
        C = trust transition hyperparameter (controls IBCF vs CBF balance)
    
    **GROUP AGGREGATION**: AVERAGE (Baseline Strategy)
    - Group score = Mean of individual user scores
    - Reference: Masthoff, J. (2011). Group recommender systems: Combining individual models.
    - This is the standard baseline for controlled comparison with H2.
    
    **KEY DIFFERENCE FROM H2**:
    - H1: Weighted hybridization (IBCF + CBF)
    - H2: Switching hybridization (UBCF + CBF)
    - Both use AVERAGE group aggregation for fair comparison.
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
        
        # **SANITY CHECK**: Ensure at least one model produces a valid score
        if np.isnan(ib_pred) and np.isnan(cb_pred):
            # Both models failed - return global mean as last resort
            return self.ib_model.global_mean
        
        # Handle cases where one model fails
        if np.isnan(ib_pred): 
            return cb_pred  # Pure CBF
        if np.isnan(cb_pred): 
            return ib_pred  # Pure IBCF
            
        # Dynamic Weighting based on Trust (neighbors count)
        n = neighbors_count
        alpha = n / (n + self.C) # Trust IB more as n increases
        beta = self.C / (n + self.C)
        
        # **SANITY CHECK**: Ensure weights sum to 1.0
        assert abs(alpha + beta - 1.0) < 1e-6, f"Weights don't sum to 1: α={alpha}, β={beta}"
        
        final_score = (alpha * ib_pred) + (beta * cb_pred)
        
        # **SANITY CHECK**: Ensure final score is valid
        assert not np.isnan(final_score), f"Final score is NaN: α={alpha}, β={beta}, IBCF={ib_pred}, CBF={cb_pred}"
        assert not np.isinf(final_score), f"Final score is Inf: α={alpha}, β={beta}, IBCF={ib_pred}, CBF={cb_pred}"
        
        return final_score

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
        Generates group recommendations using AVERAGE aggregation strategy.
        
        **ALGORITHM**:
        1. Filter out items watched by any group member (in training set)
        2. For each candidate item:
           - Predict individual score for each user (using weighted hybrid)
           - Aggregate using AVERAGE strategy
        3. Rank by group score and return top-K
        
        **GROUP AGGREGATION**: AVERAGE (Baseline)
        - Group_Score(i) = Mean([score(u, i) for u in group])
        - This is the ONLY aggregation strategy used in H1.
        - Reference: Masthoff, J. (2011).
        
        **IMPORTANT**: 
        - This is NOT least misery, harmonic mean, or fairness-aware.
        - This is a conscious baseline choice for controlled comparison with H2.
        - Both H1 and H2 use AVERAGE aggregation.
        
        Args:
            user_ids: List of user IDs in the group
            candidates: List of candidate movie IDs (must be in CF matrix)
            top_k: Number of recommendations to return
            
        Returns:
            List[Dict]: Recommendations with movie_id, score, explanations
        """
        # **TYPE ENFORCEMENT**: Ensure all IDs are integers
        candidates = [int(mid) for mid in candidates]
        user_ids = [int(uid) for uid in user_ids]
        
        # Filter out movies watched by any member
        # ------------------------------------------------------------------
        # OPTIMIZED: Pre-fetch watched items for all group members
        # ------------------------------------------------------------------
        group_watched_items = set()
        for uid in user_ids:
            try:
                if uid in self.ib_model.raw_um.index:
                    # Fast Pandas: Get indices of non-null values (watched items)
                    # accessing .loc[uid] once is much faster than .loc[uid, mid] N times
                    user_series = self.ib_model.raw_um.loc[uid]
                    watched_mids = user_series[user_series.notna()].index.tolist()
                    group_watched_items.update([int(m) for m in watched_mids])
            except Exception as e:
                print(f"[WARNING] Error fetching history for user {uid}: {e}")
                continue
                
        # Filter candidates (Set Difference)
        valid_candidates = [mid for mid in candidates if mid not in group_watched_items]
        
        # Filter sequels to prevent spoilers (unless prequel is seen)
        try:
            valid_candidates = self._filter_sequels(valid_candidates, user_ids)
        except Exception as e:
            print(f"[WARNING] Sequel filtering failed: {e}. Continuing without sequel filter.")
                
        # **GROUP AGGREGATION: AVERAGE STRATEGY**
        # Batch Predict for all users and candidates at once (Vectorized Optimization)
        print(f"    [PREDICT] Batch processing {len(valid_candidates)} candidates...", flush=True)
        
        try:
            # 1. IB Batch
            ib_batch = self.ib_model.predict_for_group(user_ids, valid_candidates)
        except AttributeError:
             # Fallback if old valid_candidates format or method missing
             ib_batch = {}
        
        try:
            # 2. CB Batch
            cb_batch = self.cb_model.predict_for_group(user_ids, valid_candidates)
        except AttributeError:
             cb_batch = {}

        group_results = []
        total_candidates = len(valid_candidates)
        
        for idx, mid in enumerate(valid_candidates, 1):
            # Progress logging every 200 (less spammy)
            if idx % 200 == 0:
                print(f"    [PREDICT] Processing movie {idx}/{total_candidates}...", flush=True)
            
            scores = []
            for uid in user_ids:
                # Retrieve pre-calculated values
                ib_res = ib_batch.get(mid, {}).get(uid)
                cb_val = cb_batch.get(mid, {}).get(uid)
                
                ib_pred = np.nan
                n = 0
                if ib_res:
                    ib_pred = ib_res.get('score', np.nan)
                    n = ib_res.get('n_neighbors', 0)
                
                cb_pred = cb_val if cb_val is not None else np.nan
                
                # Hybrid Logic (Matches self.predict)
                # -----------------------------------
                
                # 1. Both Failed
                if np.isnan(ib_pred) and np.isnan(cb_pred):
                    # Use global mean if available
                    if hasattr(self.ib_model, 'global_mean'):
                         s = self.ib_model.global_mean
                    else:
                         continue # Skip if completely unknown
                
                # 2. One Failed
                elif np.isnan(ib_pred):
                     s = cb_pred
                elif np.isnan(cb_pred):
                     s = ib_pred
                
                # 3. Both Succeeded
                else: 
                     # Dynamic Weighting
                     alpha = n / (n + self.C)
                     beta = self.C / (n + self.C)
                     s = (alpha * ib_pred) + (beta * cb_pred)
                
                if not np.isnan(s):
                    scores.append(s)
            
            if scores:
                # **AVERAGE AGGREGATION** (Baseline Strategy)
                # Group score = arithmetic mean of individual user scores
                final_score = np.mean(scores)
                group_results.append((mid, final_score))
                
        group_results.sort(key=lambda x: x[1], reverse=True)
        top_items = group_results[:top_k]
        
        # Generate Explanations for Top Items
        print(f"    [PREDICT] Generating explanations for top {top_k} recommendations...", flush=True)
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
                'movie_id': int(mid),  # **ENFORCE INTEGER**
                'score': float(score),
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
