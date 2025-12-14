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
        Predicts rating for a user on a movie.
        """
        # 1. Get IB Prediction & Info
        ib_pred, info = self.ib_model.predict(user_id, movie_id, return_info=True)
        neighbors_count = info.get('n_neighbors', 0)
        
        # 2. Get CB Prediction
        # CB is the fallback / safety net
        cb_pred = self.cb_model.predict_rating(user_id, movie_id)
        
        # 3. Hybrid Logic
        
        # Case A: IB is impossible (NaN) -> Use CB
        if np.isnan(ib_pred):
            return cb_pred
            
        # Case B: CB is impossible (NaN) -> Use IB (rare)
        if np.isnan(cb_pred):
            return ib_pred
            
        # Case C: Both available -> Weighted Average
        n = neighbors_count
        
        # Weight for IB
        alpha = n / (n + self.C)
        
        # Weight for CB
        beta = self.C / (n + self.C)
        
        final_score = (alpha * ib_pred) + (beta * cb_pred)
        
        return final_score

    def explain(self, user_id, movie_id):
        """
        Generates individual explanations using strict sentence templates.
        Selection based on evidence strength (IBCF vs CB).
        """
        response = {'type': 'Hybrid', 'final_reason': "This recommendation could not be explained due to insufficient similarity.", 'evidence': {}}
        
        # --- 1. IBCF Candidate ---
        ib_sentence = None
        ib_evidence_score = 0
        valid_ib_traits = {}
        
        ib_expl = self.ib_model.get_explanation(user_id, movie_id)
        neighbors = ib_expl.get('neighbors', [])
        valid_neighbors = []
        
        SIM_THRESHOLD = 0.1
        
        for n in neighbors:
            if n.get('similarity', 0) < SIM_THRESHOLD: continue
            
            # Check Evidence Rule: Share >=1 Genre OR >=2 Keywords
            # n has keys: 'id', 'title', 'rating', 'similarity'
            neighbor_id = n.get('id')
            if not neighbor_id: continue
            
            traits = self.cb_model.get_shared_traits(movie_id, neighbor_id)
            
            has_genre = len(traits.get('Genres', [])) >= 1
            has_keywords = len(traits.get('Keywords', [])) >= 2
            
            if has_genre or has_keywords:
                valid_neighbors.append((n, traits))
                
            if len(valid_neighbors) >= 2: break
            
        if valid_neighbors:
            ib_evidence_score = len(valid_neighbors)
            # Generate Sentence
            # Use top neighbor
            top_n, top_traits = valid_neighbors[0]
            valid_ib_traits = top_traits
            
            reasons = []
            if top_traits.get('Genres'):
                reasons.append(f"{list(top_traits['Genres'])[0].lower()} genre") # "comedy genre"
            elif top_traits.get('Keywords'):
                reasons.append("themes")
            
            reason_str = "similar content"
            if reasons:
                # specific: "share the comedy genre"
                reason_str = f"share the {reasons[0]}"
            
            ib_sentence = f"This movie is recommended because you liked {top_n['title']}, and both movies {reason_str}."

        # --- 2. CB Candidate ---
        cb_sentence = None
        cb_evidence_score = 0
        valid_cb_traits = {}
        
        # Get history >= 4.0
        try:
            # Check if user exists in raw_um
            if user_id in self.ib_model.raw_um.index:
                user_hist = self.ib_model.raw_um.loc[user_id]
                liked_items = user_hist[user_hist >= 4.0].index.tolist()
                
                # Find overlap
                max_features = 0
                best_match_id = None
                
                for hist_mid in liked_items:
                    traits = self.cb_model.get_shared_traits(movie_id, hist_mid)
                    # Evidence: Count Genres + Directors + Actors
                    count = len(traits.get('Genres', [])) + len(traits.get('Directors', [])) + len(traits.get('Actors', []))
                    if count > max_features:
                        max_features = count
                        valid_cb_traits = traits
                        best_match_id = hist_mid
                
                if max_features > 0:
                    cb_evidence_score = max_features
                    # Generate Sentence
                    # "matches your preferences because it shares [Feature]..."
                    feat_names = []
                    if valid_cb_traits:
                        parts = []
                        if valid_cb_traits.get('Genres'):
                            g_list = [g.lower() for g in list(valid_cb_traits['Genres'])[:3]]
                            if len(g_list) > 1:
                                parts.append(f"the {', '.join(g_list[:-1])} and {g_list[-1]} genres")
                            else:
                                parts.append(f"the {g_list[0]} genre")
                        else:
                            parts.append("genres")

                        if valid_cb_traits.get('Directors'): parts.append("directors")
                        if valid_cb_traits.get('Actors'): parts.append("actors")
                        
                        fea_str = " and ".join(parts)
                        cb_sentence = f"This movie matches your preferences because it shares {fea_str} with movies you rated highly."
        except Exception:
            pass # Fallback if history access fails

        # --- 3. Selection ---
        
        selected = None
        
        if ib_sentence and cb_sentence:
            if ib_evidence_score >= 1:
                selected = "IBCF"
            else:
                selected = "CB"
        elif ib_sentence:
            selected = "IBCF"
        elif cb_sentence:
            selected = "CB"
            
        if selected == "IBCF":
            response['final_reason'] = ib_sentence
            response['evidence'] = valid_ib_traits
        elif selected == "CB":
            response['final_reason'] = cb_sentence
            response['evidence'] = valid_cb_traits
            
        return response

    def recommend_for_group(self, user_ids, candidates, top_k=10):
        """
        Generates group recommendations.
        1. Filters out movies watched by ANY member.
        2. Aggregates scores (Average Strategy).
        3. Returns Top-K with explanations.
        """
        # 1. Filter Watched Items

        valid_candidates = []
        
        for mid in candidates:
            watched = False
            for uid in user_ids:
                if uid in self.ib_model.raw_um.index and mid in self.ib_model.raw_um.columns:
                    if pd.notna(self.ib_model.raw_um.loc[uid, mid]):
                        watched = True
                        break
            
            if not watched:
                valid_candidates.append(mid)
        
        # 1.5 Filter Sequels (If Prequel not watched)
        valid_candidates = self._filter_sequels(valid_candidates, user_ids)
                
        # 2. Score Candidates
        group_results = []
        
        for mid in valid_candidates:
            scores = []
            for uid in user_ids:
                s = self.predict(uid, mid)
                if not np.isnan(s):
                    scores.append(s)
            
            if scores:
                avg_score = np.mean(scores)
                group_results.append((mid, avg_score))
                
        # 3. Sort and Explain
        group_results.sort(key=lambda x: x[1], reverse=True)
        top_items = group_results[:top_k]
        
        final_recommendations = []
        for mid, score in top_items:
            explanations = {uid: self.explain(uid, mid) for uid in user_ids}
            
            # Generate Group Narrative
            group_reason = self._generate_group_explanation(explanations)
            
            final_recommendations.append({
                'movie_id': mid,
                'score': score,
                'group_explanation': group_reason,
                'explanations': explanations
            })
            
        return final_recommendations

    def _generate_group_explanation(self, explanations):
        """
        Synthesizes a group-level explanation.
        STRICT FORMAT: 
        - Dominant component
        - Common patterns
        - Explanation sentence
        """
        from collections import Counter
        import re
        
        # 1. Determine Group Dominant Component
        comp_counts = Counter()
        for expl in explanations.values():
            comp_counts[expl.get('dominant_component', 'CB')] += 1
        
        # Majority wins
        if comp_counts:
            group_dominant = comp_counts.most_common(1)[0][0]
        else:
            group_dominant = "CB"
        
        # 2. Extract Patterns
        all_genres = []
        all_themes = []
        
        for expl in explanations.values():
            text = expl.get('final_reason', '')
            # Regex for "share the X genre" (IBCF) or "shares X, Y ... with movies" (CB)
            
            # IBCF Pattern: "share the [X] genre"
            match = re.search(r"share the (.*?) genre", text, re.IGNORECASE)
            if match:
                g = match.group(1).strip()
                all_genres.append(g)
                
            # CB Pattern: "shares [X, Y] with movies"
            match_cb = re.search(r"shares (.*?) with movies", text, re.IGNORECASE)
            if match_cb:
                content = match_cb.group(1)
                pass

            # Check 'evidence' dict for actual values!
            evidence = expl.get('evidence', {})
            if evidence.get('Genres'):
                all_genres.extend(list(evidence['Genres']))
            if evidence.get('Keywords'):
                all_themes.extend(list(evidence['Keywords']))
            
        # 3. Analyze Patterns
        common_patterns = []
        
        # Tally
        g_counts = Counter(all_genres)
        t_counts = Counter(all_themes)
        
        top_genres = [g for g, c in g_counts.most_common(3) if c > 0]
        top_themes = [t for t, c in t_counts.most_common(3) if c > 0]
        
        if top_genres:
            common_patterns.append(f"Genres: {', '.join(top_genres)}")
        if top_themes:
            common_patterns.append(f"Themes: {', '.join(top_themes)}")
            
        if not common_patterns:
            common_patterns.append("Diverse individual preferences")

        # 4. Construct Sentence
        explanation_sentence = ""
        if top_genres:
            explanation_sentence = f"This movie fits the group because most members frequently watch {', '.join(top_genres[:2])} movies."
        elif top_themes:
            explanation_sentence = f"The group shares a thematic interest in {', '.join(top_themes[:2])}."
        else:
            explanation_sentence = "Recommended to satisfy individual high-rating predictions despite lacking a unified group theme."

        # 5. Final Formatting
        lines = []
        lines.append(f"Group Insight:")
        lines.append(f"  - Common patterns: {'; '.join(common_patterns)}")
        lines.append(f"  - Explanation: {explanation_sentence}")
        
        return "\n".join(lines)

    def _filter_sequels(self, candidates, user_ids):
        """
        Removes candidates that are sequels if the user hasn't seen the prequel.
        Heuristic: 
        1. Find "Series" by Title Stem (first 3 words). 
        2. Sort by Year.
        3. If strict sequence, enforce history check.
        """
        # Load movies with year (Assume self.cb_model.movies_df has 'year' now)
        movies = self.cb_model.movies_df
        
        # Identify watched set (union of group)
        watched_ids = set()
        for uid in user_ids:
            if uid in self.ib_model.raw_um.index:
                user_hist = self.ib_model.raw_um.loc[uid]
                watched_ids.update(user_hist[user_hist.notna()].index.tolist())
                
        # Candidate Metadata
        cand_meta = movies[movies['movieId'].isin(candidates)].copy()
        
        # 1. Group by Stem (First 2-3 words simple heuristic)
        # "Star Wars: Episode I" -> "Star Wars:"
        def get_stem(title):
            words = title.split()
            if len(words) > 2:
                return " ".join(words[:2]).lower() # conservative (2 words)
            return title.lower()
            
        cand_meta['stem'] = cand_meta['title'].apply(get_stem)
        
        # Find all movies in DB that match these stems (to find prequels even if not in candidates)
        # Optimization: Just scan all movies? 12k is small.
        stems = set(cand_meta['stem'].unique())
        
        # Get all related movies
        # Apply stem on all movies? Slow? 
        # Vectorized string ops are fast enough for 12k.
        all_movies = movies.copy()
        all_movies['stem'] = all_movies['title'].apply(get_stem)
        
        related = all_movies[all_movies['stem'].isin(stems)].sort_values(['stem', 'year'])
        
        # Identify invalid candidates
        invalid_candidates = set()
        
        for stem, group in related.groupby('stem'):
            if len(group) < 2: continue
            
            # Sorted by Year
            sorted_mids = group['movieId'].values
            
            # Check prerequisites
            for i in range(1, len(sorted_mids)):
                current_mid = sorted_mids[i]
                prev_mid = sorted_mids[i-1]
                
                # If current is in candidates
                if current_mid in candidates:
                    # Logic: If prev_mid NOT in watched_ids -> Block current_mid
                    if prev_mid not in watched_ids:
                        # Double check year diff (don't block remakes if same year, but we deduplicated)
                        # Also check title similarity?
                        invalid_candidates.add(current_mid)
                        # Propagate block? (If blocked 2, block 3?)
                        # Yes, if 2 is invalid, 3 (which needs 2) is also invalid.
                        # Since we check `prev_mid not in watched`, but if prev_mid was blocked candidate?
                        # `watched_ids` is strictly history. So yes, propagate.
                        
        valid = [c for c in candidates if c not in invalid_candidates]
        return valid
