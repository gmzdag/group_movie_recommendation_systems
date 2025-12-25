"""
Structured Output Generator for Group Movie Recommendations

This module produces TEXT-BASED, STRUCTURED outputs for three distinct sections:
- SECTION A: Top-10 Ranked Group Recommendations
- SECTION B: Common Watchlist (Unranked, Explicit)
- SECTION C: Shared Group Interest Themes

All outputs are deterministic, UI-agnostic, and follow strict explanation rules.

NEW: Includes temporal preference filtering to ensure recommendations match
users' historical viewing patterns regarding movie release years.
"""

import re
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Set, Tuple, Optional
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from src.recommender.temporal_preference_analyzer import TemporalPreferenceAnalyzer

try:
    from src.agents.film_curator import FilmFilterAgent, get_filter_agent
    HAS_AGENT = True
except ImportError:
    HAS_AGENT = False


class StructuredOutputGenerator:
    """
    Generates structured, text-based recommendation outputs.
    
    This class does NOT design UI, does NOT mention frontend/layout.
    It produces JSON-serializable data structures ready for UI consumption.
    """
    
    def __init__(self, hybrid_model_1, hybrid_model_2, hybrid_model_3, 
                 movies_df, watchlist_df, cf_matrix, ratings_df=None,
                 enable_temporal_filtering=True, use_multi_model_selection=True,
                 film_agent=None):  # Added film_agent parameter
        """
        Args:
            hybrid_model_1: HybridModel1 instance (Dynamic Weighted Hybrid)
            hybrid_model_2: SwitchingHybridRecommender instance
            hybrid_model_3: WatchlistHybridModel instance
            movies_df: DataFrame with movie metadata
            watchlist_df: DataFrame with user watchlists
            cf_matrix: Collaborative filtering matrix (user-item ratings)
            ratings_df: DataFrame with ratings (userId, movieId, rating, timestamp)
            enable_temporal_filtering: If True, apply temporal preference filtering
            use_multi_model_selection: If True, use multi-model selection for Section A
            film_agent: Optional FilmFilterAgent instance for "I'll guide you" flow
        """
        self.h1 = hybrid_model_1
        self.h2 = hybrid_model_2
        self.h3 = hybrid_model_3
        self.movies_df = movies_df
        self.watchlist_df = watchlist_df
        self.cf_matrix = cf_matrix
        self.enable_temporal_filtering = enable_temporal_filtering
        self.use_multi_model_selection = use_multi_model_selection
        self.film_agent = film_agent
        
        # Initialize temporal analyzer if ratings data is provided
        if ratings_df is not None and enable_temporal_filtering:
            self.temporal_analyzer = TemporalPreferenceAnalyzer(ratings_df, movies_df)
        else:
            self.temporal_analyzer = None
        
        # Load model performance for multi-model selection
        self.model_performance = self._load_model_performance() if use_multi_model_selection else None
        
    def generate_three_section_output(self, group_users: List[int], user_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Generates all three sections for a group.
        
        Args:
            group_users: List of user IDs
            user_prompt: Optional user text prompt for AI guidance
        
        Returns:
            {
                'section_a_top_recommendations': [...],
                'section_b_common_watchlist': [...],
                'section_c_shared_interests': [...]
            }
        """
        # Get watched movies (global exclusion set)
        watched_set = self._get_watched_set(group_users)
        
        # Get direct watchlist matches (for exclusion from Section A)
        direct_watchlist_set = self._get_direct_watchlist_set(group_users)
        
        # ============================================================================
        # GLOBAL FILTERING (AGENT)
        # ============================================================================
        valid_ids_set = None
        filtered_candidates_list = None
        
        if user_prompt and HAS_AGENT and self.film_agent:
            print(f"[StructuredOutput] Applying Global Agent Filter for prompt: '{user_prompt}'")
            # 1. Get ALL movies as candidates (Global Scan)
            # Function default is limit=1000, we override with scan_all=True to get EVERYTHING
            all_candidates = self._get_raw_enriched_candidates(
                group_users, watched_set, direct_watchlist_set, limit=100000, scan_all=True
            )
            
            # 2. Run Agent
            filter_result = self.film_agent.filter_from_prompt(user_prompt, all_candidates)
            valid_ids_list = filter_result.get('filtered_movie_ids', [])
            valid_ids_set = set(valid_ids_list)
            
            # 3. Create filtered candidates list for Section A
            filtered_candidates_list = [c for c in all_candidates if c['movieId'] in valid_ids_set]
            
            # Capture agent output for reasoning augmentation later if needed
            # (Reasoning is attached in _apply_agent_filter, but here we do manual)
            agent_rec = filter_result.get('agent_recommendation')
            agent_reasons = {c['movieId']: filter_result.get('filters_applied', {}) for c in filtered_candidates_list}
            
            print(f"[StructuredOutput] Global Filter kept {len(valid_ids_set)} / {len(all_candidates)} movies.")

        # ============================================================================
        # GENERATE SECTIONS
        # ============================================================================
        
        # Section A
        if self.use_multi_model_selection and self.model_performance:
            section_a = self._generate_section_a_multi_model(
                group_users, watched_set, direct_watchlist_set, 
                user_prompt=user_prompt,
                citation_reasoning=agent_reasons if filtered_candidates_list else None,
                pre_filtered_candidates=filtered_candidates_list # Pass PRE-FILTERED list
            )
        else:
            section_a = self._generate_section_a(
                group_users, watched_set, direct_watchlist_set, 
                user_prompt=user_prompt,
                citation_reasoning=agent_reasons if filtered_candidates_list else None,
                pre_filtered_candidates=filtered_candidates_list
            )
        
        # Section B (Apply global filter if valid_ids_set exists)
        section_b = self._generate_section_b(
            group_users, watched_set, section_a,
            allowed_ids=valid_ids_set 
        )
        
        # Section C (Apply global filter if valid_ids_set exists)
        section_c = self._generate_section_c(
            group_users, watched_set, section_a, section_b,
            allowed_ids=valid_ids_set
        )
        
        # Section D (Apply global filter if valid_ids_set exists)
        section_d = self._generate_section_d(
            group_users, watched_set, section_a, section_b,
            allowed_ids=valid_ids_set
        )
        
        # Section E - Hybrid 1 specific recommendations
        section_e = self._generate_section_e(
            group_users, watched_set, section_a, section_b, section_d,
            allowed_ids=valid_ids_set
        )
        
        # Section F - Hybrid 2 specific recommendations
        section_f = self._generate_section_f(
            group_users, watched_set, section_a, section_b, section_d,
            allowed_ids=valid_ids_set
        )
        
        # ============================================================================
        # TMDB ENRICHMENT: Add poster/backdrop/trailer URLs
        # ============================================================================
        from src.utils.tmdb_enrichment import enrich_movies_batch
        
        # Collect all unique movies from all sections
        all_movies = []
        
        # Section A movies
        for movie in section_a:
            all_movies.append({
                'movie_id': movie['movie_id'],
                'title': movie['title']
            })
        
        # Section B movies
        for item in section_b:
            all_movies.append({
                'movie_id': item['movie_id'],
                'title': item['title']
            })
        
        # Section C movies (nested in themes)
        for theme in section_c:
            for movie in theme.get('recommended_movies', []):
                all_movies.append({
                    'movie_id': movie['movie_id'],
                    'title': movie['title']
                })
        
        # Section D movies
        for item in section_d:
            all_movies.append({
                'movie_id': item['movie_id'],
                'title': item['title']
            })
        
        # Section E movies
        for item in section_e:
            all_movies.append({
                'movie_id': item['movie_id'],
                'title': item['title']
            })
        
        # Section F movies
        for item in section_f:
            all_movies.append({
                'movie_id': item['movie_id'],
                'title': item['title']
            })
        
        # Remove duplicates (keep unique movie_ids)
        unique_movies = {m['movie_id']: m for m in all_movies}.values()
        
        # Enrich with TMDB data (parallel fetching with caching)
        enriched_movies = enrich_movies_batch(list(unique_movies))
        
        # Create lookup dict: movie_id -> {poster_url, backdrop_url, trailer_url}
        tmdb_lookup = {
            m['movie_id']: {
                'poster_url': m.get('poster_url'),
                'backdrop_url': m.get('backdrop_url'),
                'trailer_url': m.get('trailer_url')
            }
            for m in enriched_movies
        }
        
        # Add TMDB data to Section A
        for movie in section_a:
            movie.update(tmdb_lookup.get(movie['movie_id'], {}))
        
        # Add TMDB data to Section B
        for item in section_b:
            item.update(tmdb_lookup.get(item['movie_id'], {}))
        
        # Add TMDB data to Section C
        for theme in section_c:
            for movie in theme.get('recommended_movies', []):
                movie.update(tmdb_lookup.get(movie['movie_id'], {}))
        
        # Add TMDB data to Section D
        for item in section_d:
            item.update(tmdb_lookup.get(item['movie_id'], {}))
        
        # Add TMDB data to Section E
        for item in section_e:
            item.update(tmdb_lookup.get(item['movie_id'], {}))
        
        # Add TMDB data to Section F
        for item in section_f:
            item.update(tmdb_lookup.get(item['movie_id'], {}))
        
        return {
            'section_a_top_recommendations': section_a,
            'section_b_common_watchlist': section_b,
            'section_c_shared_interests': section_c,
            'section_d_watchlist_inspired': section_d,
            'section_e_hybrid1_picks': section_e,
            'section_f_hybrid2_picks': section_f
        }
    
    # ========================================================================
    # SECTION A: TOP-10 RANKED GROUP RECOMMENDATIONS
    # ========================================================================
    
    def _load_model_performance(self) -> Optional[Dict[str, Dict]]:
        """Load model performance metrics from cache."""
        import json
        import os
        
        cache_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "data", "cache", "model_performance.json"
        )
        
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    performance = json.load(f)
                return performance
            except Exception as e:
                print(f"[WARNING] Could not load model performance: {e}")
                return None
        else:
            return None
    
    def _calculate_quota_allocation(self, total_slots: int = 10) -> Dict[str, int]:
        """
        Calculate slot allocation for each model based on performance ranking.
        """
        if not self.model_performance:
            # Fallback: equal distribution
            return {'h1': 4, 'h2': 3, 'h3': 3}
        
        # Get NDCG scores
        scores = {
            'h1': self.model_performance['h1'].get('ndcg@10', 0.0),
            'h2': self.model_performance['h2'].get('ndcg@10', 0.0),
            'h3': self.model_performance['h3'].get('ndcg@10', 0.0)
        }
        
        total_score = sum(scores.values())
        
        if total_score < 0.01:
            # All models failed, use equal distribution
            return {'h1': 4, 'h2': 3, 'h3': 3}
        
        # Proportional allocation
        allocation = {}
        for model, score in scores.items():
            allocation[model] = max(1, int(round((score / total_score) * total_slots)))
        
        # Adjust to exactly total_slots
        current_total = sum(allocation.values())
        if current_total != total_slots:
            # Give extra slots to best model or remove from worst
            best_model = max(scores.items(), key=lambda x: x[1])[0]
            allocation[best_model] += (total_slots - current_total)
        
        return allocation
    
    def _get_raw_enriched_candidates(self, group_users: List[int], 
                                 watched_set: Set[int], 
                                 direct_watchlist_set: Set[int], 
                                 limit: int = 1000,
                                 scan_all: bool = False) -> List[Dict[str, Any]]:
        """
        Generate a list of raw candidates enriched with metadata for Agent Filtering.
        
        Args:
            scan_all: If True, returns ALL movies in the database (for strict filtering).
                      If False, returns only CF/Popularity candidates (for relevance).
        """
        enriched_candidates = []
        
        if scan_all:
            print("[StructuredOutput] Global Search Mode Enabled: Scanning entire database for Agent.")
            # global search: use all movies (excluding watched/watchlist)
            excluded = watched_set | direct_watchlist_set
            # Only keep movies NOT in excluded
            rows = self.movies_df[~self.movies_df['movieId'].isin(excluded)]
        else:
            # Standard candidate generation
            candidate_ids = list(set(self._get_candidates_for_section_a(group_users, watched_set, direct_watchlist_set)))
            
            # Limit if too many
            if len(candidate_ids) > limit:
                candidate_ids = candidate_ids[:limit]
                
            rows = self.movies_df[self.movies_df['movieId'].isin(candidate_ids)]
        
        # Ensure we don't have NaNs in movieId
        rows = rows.dropna(subset=['movieId'])
        
        # Convert to dictionary format required by Agent
        raw_dicts = rows.to_dict('records')
        
        for row in raw_dicts:
            enriched_candidates.append({
                'movieId': int(row['movieId']),
                'title': str(row.get('title', 'Unknown')),
                'year': int(row.get('year', 0)) if pd.notna(row.get('year')) else 0,
                'genres': str(row.get('genres', '')).split('|'),
                'overview': str(row.get('overview', '')),
                'keywords': str(row.get('keywords', '')),
                'actors': str(row.get('actors', '')),
                'director': str(row.get('director', '')),
                'countries': str(row.get('Production_Countries', '')),
                'runtime': int(row.get('runtime', 0)) if pd.notna(row.get('runtime')) else 0,
                'vote_average': float(row.get('vote_average', 0.0)) if pd.notna(row.get('vote_average')) else 0.0
            })
        
        return enriched_candidates

    def _apply_agent_filter(self, user_prompt: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Use the FilmFilterAgent to filter and rank candidates.
        """
        if not self.film_agent:
            print("[WARNING] Agent requested but not initialized.")
            return []
            
        print(f"[AGENT] Filtering {len(candidates)} candidates with prompt: '{user_prompt}'")
        
        try:
            # Call agent
            filter_result = self.film_agent.filter_from_prompt(
                user_prompt=user_prompt,
                candidate_movies=candidates
            )
            
            filtered_ids = set(filter_result.get('filtered_movie_ids', []))
            agent_rec = filter_result.get('agent_recommendation')

            # SAFETY: If agent found a specific movie, ensure it's in the filtered set
            if agent_rec and agent_rec.get('movieId'):
                filtered_ids.add(agent_rec.get('movieId'))
            
            # Filter the candidates list based on agent's selection
            final_list = [c for c in candidates if c['movieId'] in filtered_ids]
            
            # If agent recommended a specific Top 1, move it to the front
            if agent_rec:
                top_id = agent_rec.get('movieId')
                # Find it (it might be in final_list now, or might need to be fetched from raw candidates)
                bg_item = next((c for c in final_list if c['movieId'] == top_id), None)
                
                # If not in filtered list but exists in candidates (Safety Fallback)
                if not bg_item:
                     bg_item = next((c for c in candidates if c['movieId'] == top_id), None)
                     if bg_item:
                         final_list.append(bg_item)
                
                if bg_item:
                    # Move to front
                    if bg_item in final_list:
                        final_list.remove(bg_item)
                    final_list.insert(0, bg_item)
                    
                    # Add agent reasoning to it
                    bg_item['agent_reasoning'] = agent_rec.get('reason', '')
                    # Ensure original explanations are accessible
                    bg_item['original_explanations'] = bg_item.get('explanations', {})
            
            return final_list
            
        except Exception as e:
            print(f"[AGENT ERROR] {e}")
            return [] # Fallback to empty (caller will handle fallback to standard logic)

    def _generate_section_a_multi_model(
        self,
        group_users: List[int],
        watched_set: Set[int],
        direct_watchlist_set: Set[int],
        user_prompt: Optional[str] = None,
        citation_reasoning: Optional[Dict] = None,
        pre_filtered_candidates: Optional[List[Dict]] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate Top-10 using multi-model selection strategy.
        If pre_filtered_candidates is provided (Agent Mode), we skip candidate generation 
        and just Rank the provided candidates.
        """
        
        # --- AGENT / STRICT FILTER PATH ---
        if pre_filtered_candidates is not None:
             # We have a strict list of allowed movies.
             # We just need to RANK them using the group model (HybridModel1) 
             # because it's the best at scoring.
             
             if not pre_filtered_candidates:
                 print("[SECTION A] Agent filter returned 0 results.")
                 return []
             
             filtered_ids = [c['movieId'] for c in pre_filtered_candidates]
             
             # Rank them by group preference
             scored_recs = self.h1.recommend_for_group(group_users, filtered_ids, top_k=10)
             
             output = []
             for rec in scored_recs:
                 mid = rec['movie_id']
                 # Construct reasoning string from parsed criteria
                 criteria = citation_reasoning.get(mid) if citation_reasoning else None
                 reason_str = f"Matches criteria: {criteria}" if criteria else "Matches your request."
                 
                 output.append({
                     'movie_id': mid,
                     'title': self._get_movie_title(mid),
                     'group_score': round(rec['score'], 2),
                     'source_model': 'AI_AGENT', 
                     'group_explanation': f"{reason_str}",
                     'signal_source': 'AI_AGENT',
                     'user_explanations': rec.get('explanations', {})
                 })
             return output

        # --- AGENT PATH (Legacy Fallback if prompt given but no pre-filter?) ---
        if user_prompt and HAS_AGENT and pre_filtered_candidates is None:
            # This branch should rarely be hit if Orchestrator works, 
            # but keeping for safety or direct calls.
            # ... (Existing logic shifted or removed?)
            # For brevity/safety, let's just warn and fall through or simplistic logic
            pass
    
        # --- STANDARD PATH ---
        
        print("\n[SECTION A] Using QUOTA-BASED MIXED ENSEMBLE strategy")
        
        # 1. Calculate how many slots each model gets (e.g., 4-3-3)
        quotas = self._calculate_quota_allocation(total_slots=10)
        print(f"[SECTION A] Slot allocation: {quotas}")
        
        # 2. Get candidates for Section A
        candidates = self._get_candidates_for_section_a(
            group_users, watched_set, direct_watchlist_set
        )
        
        if not candidates:
            return []
            
        # 3. Collect top picks from each model based on their quota
        combined_picks = []
        seen_mids = set()
        
        # We iterate through models to fill the slots
        for m_key, quota in quotas.items():
            model = getattr(self, m_key)
            m_label = m_key.upper()
            
            # Request more than quota to handle duplicates from previous models
            model_recs = model.recommend_for_group(group_users, candidates, top_k=quota + 5)
            
            added_from_this_model = 0
            for rec in model_recs:
                mid = rec['movie_id']
                if mid not in seen_mids and added_from_this_model < quota:
                    rec['source_model'] = m_label
                    combined_picks.append(rec)
                    seen_mids.add(mid)
                    added_from_this_model += 1
        
        # 4. Fill remaining slots if any (due to low candidates or high overlap)
        if len(combined_picks) < 10:
            remaining = 10 - len(combined_picks)
            # Fallback to Hybrid 1 (most stable) for extra picks
            extra_recs = self.h1.recommend_for_group(group_users, candidates, top_k=20)
            for rec in extra_recs:
                mid = rec['movie_id']
                if mid not in seen_mids and len(combined_picks) < 10:
                    rec['source_model'] = 'H1_EXTRA'
                    combined_picks.append(rec)
                    seen_mids.add(mid)

        # NEW: Global Re-sort of the mixed ensemble to ensure highest scores are on top
        combined_picks.sort(key=lambda x: x['score'], reverse=True)

        # 5. Format results
        section_a_output = []
        for rec in combined_picks:
            movie_id = rec['movie_id']
            title = self._get_movie_title(movie_id)
            
            # Use model-provided explanation or fallback
            group_expl = rec.get('group_explanation', 'High-match recommendation based on group tastes.')
            
            # Filter explanations (no direct watchlist)
            user_explanations = {}
            # SAFETY: Check if explanations exists and is a dict
            if rec.get('explanations') and isinstance(rec['explanations'], dict):
                for uid, expl in rec['explanations'].items():
                    if not self._is_direct_watchlist_explanation(expl):
                        user_explanations[uid] = expl
            
            # TEMP FIX: Allow empty explanations\r\n
            
            # if not user_explanations:\r\n
            
            #     continue
            
            # Get full metadata
            metadata = self._get_movie_metadata(movie_id)
            
            section_a_output.append({
                'movie_id': movie_id,
                'title': title,
                'group_score': round(rec['score'], 2),
                'source_model': rec['source_model'],
                'model_score': round(rec['score'], 2),
                'group_explanation': group_expl,
                'signal_source': self._get_dominant_signal_source(user_explanations),
                'user_explanations': user_explanations,
                # Metadata fields
                'genres': metadata.get('genres'),
                'Overview': metadata.get('Overview'),
                'overview': metadata.get('overview'),  # Fallback
                'Director': metadata.get('Director'),
                'director': metadata.get('director'),  # Fallback
                'Actors': metadata.get('Actors'),
                'actors': metadata.get('actors'),  # Fallback
                'Production_Countries': metadata.get('Production_Countries'),
                'production_countries': metadata.get('production_countries'),  # Fallback
                'release_date': metadata.get('release_date'),
                'poster_url': metadata.get('poster_url'),
                'backdrop_url': metadata.get('backdrop_url'),
                'trailer_url': metadata.get('trailer_url')
            })
        
        return section_a_output[:10]
    
    def _generate_section_a(self, group_users: List[int], 
                           watched_set: Set[int],
                           direct_watchlist_set: Set[int],
                           user_prompt: Optional[str] = None,
                           citation_reasoning: Optional[Dict] = None,
                           pre_filtered_candidates: Optional[List[Dict]] = None) -> List[Dict[str, Any]]:
        """
        Generate Top-10 ranked group recommendations (Legacy Single Model).
        Unified with pre-filtered logic.
        """
        
        if pre_filtered_candidates is not None:
             # Reuse logic from multi-model function for agent (it's model agnostic)
             if not pre_filtered_candidates: return []
             
             filtered_ids = [c['movieId'] for c in pre_filtered_candidates]
             scored_recs = self.h1.recommend_for_group(group_users, filtered_ids, top_k=10)
             
             output = []
             for rec in scored_recs:
                 mid = rec['movie_id']
                 criteria = citation_reasoning.get(mid) if citation_reasoning else None
                 reason_str = f"Matches criteria: {criteria}" if criteria else "Matches request"
                 
                 output.append({
                     'movie_id': mid,
                     'title': self._get_movie_title(mid),
                     'group_score': round(rec['score'], 2),
                     'source_model': 'AI_AGENT',
                     'group_explanation': f"{reason_str}",
                     'signal_source': 'AI_AGENT',
                     'user_explanations': rec.get('explanations', {})
                 })
             return output

        # --- STANDARD PATH ---
        
        # 1. Generate candidates
        candidates = self._get_candidates_for_section_a(
            group_users, watched_set, direct_watchlist_set
        )
        
        if not candidates:
            return []
        
        # 2. Use Hybrid Model 1 for recommendations
        recommendations = self.h1.recommend_for_group(
            group_users, candidates, top_k=10
        )
        
        # 3. Post-process
        section_a_output = []
        
        for rec in recommendations:
            movie_id = rec['movie_id']
            
            if movie_id in direct_watchlist_set:
                continue
            
            title = self._get_movie_title(movie_id)
            
            user_explanations = {}
            # SAFETY: Check if explanations exists and is a dict
            if rec.get('explanations') and isinstance(rec['explanations'], dict):
                for uid, expl in rec['explanations'].items():
                    if self._is_direct_watchlist_explanation(expl):
                        continue
                    user_explanations[uid] = expl
            
            # TEMP FIX: Allow empty explanations\r\n
            
            # if not user_explanations:\r\n
            
            #     continue
            
            section_a_output.append({
                'movie_id': movie_id,
                'title': title,
                'group_score': round(rec['score'], 2),
                'group_explanation': rec['group_explanation'],
                'signal_source': self._get_dominant_signal_source(user_explanations),
                'user_explanations': user_explanations
            })
        
        return section_a_output[:10]
    
    def _get_candidates_for_section_a(self, group_users: List[int],
                                      watched_set: Set[int],
                                      direct_watchlist_set: Set[int]) -> List[int]:
        """
        Generate candidate movies for Section A.
        """
        candidates = set()
        
        # 1. IBCF candidates
        for uid in group_users:
            if uid in self.cf_matrix.index:
                user_ratings = self.cf_matrix.loc[uid].dropna()
                top_rated = user_ratings[user_ratings >= 4.0].sort_values(
                    ascending=False
                ).head(10)
                
                for movie_id in top_rated.index:
                    if hasattr(self.h1.ib_model, 'neighbors'):
                        neighbors_dict = self.h1.ib_model.neighbors.get(movie_id, {})
                        neighbors = list(neighbors_dict.keys())[:10]
                        candidates.update(neighbors)
        
        # 2. Watchlist similarity candidates (NOT direct matches)
        for uid in group_users:
            user_wl = self.watchlist_df[
                self.watchlist_df['userId'] == uid
            ]['movieId'].tolist()
            
            for wl_movie in user_wl:
                # 2a. Collaborative neighbors (IBCF)
                if hasattr(self.h1.ib_model, 'neighbors'):
                    neighbors_dict = self.h1.ib_model.neighbors.get(wl_movie, {})
                    similar_movies = list(neighbors_dict.keys())[:5]
                    candidates.update([m for m in similar_movies if m != wl_movie])
                
                # 2b. Content neighbors (CBF) - NEW: Matches logic in Section D
                if wl_movie in self.h3.cb_model.movie_to_idx:
                    wl_idx = self.h3.cb_model.movie_to_idx[wl_movie]
                    wl_vec = self.h3.cb_model.tfidf_matrix[wl_idx]
                    all_sims = cosine_similarity(wl_vec, self.h3.cb_model.tfidf_matrix).flatten()
                    top_indices = np.argsort(all_sims)[-11:-1][::-1] # Get top 10
                    
                    idx_to_movie_temp = {v: k for k, v in self.h3.cb_model.movie_to_idx.items()}
                    for idx in top_indices:
                        if idx in idx_to_movie_temp:
                             candidates.add(idx_to_movie_temp[idx])
        
        # 3. Popular baseline
        popular = self.cf_matrix.count().sort_values(
            ascending=False
        ).head(200).index.tolist()
        candidates.update(popular)
        
        # 4. Filter exclusions
        candidates = [
            c for c in candidates 
            if c not in watched_set and c not in direct_watchlist_set
        ]
        
        # 5. Apply temporal filtering
        if self.temporal_analyzer is not None:
            temporally_filtered = []
            for movie_id in candidates:
                is_compatible = True
                for uid in group_users:
                    if not self.temporal_analyzer.is_movie_temporally_compatible(
                        uid, movie_id, strict=False
                    ):
                        is_compatible = False
                        break
                
                if is_compatible:
                    temporally_filtered.append(movie_id)
            
            if len(temporally_filtered) >= 20:
                candidates = temporally_filtered
        
        return candidates
    
    def _is_direct_watchlist_explanation(self, explanation: Dict[str, Any]) -> bool:
        primary = explanation.get('primary_reason', '')
        if 'This movie is in your watchlist' in primary:
            return True
        if 'Direct Match' in primary:
            return True
        return False
    
    def _get_dominant_signal_source(self, user_explanations: Dict[int, Dict]) -> str:
        sources = [
            expl.get('signal_source', 'Unknown') 
            for expl in user_explanations.values()
        ]
        if not sources:
            return 'Unknown'
        counter = Counter(sources)
        return counter.most_common(1)[0][0]
    
    # ========================================================================
    # SECTION B: COMMON WATCHLIST (UNRANKED, EXPLICIT)
    # ========================================================================
    
    def _generate_section_b(self, group_users: List[int],
                           watched_set: Set[int],
                           section_a: List[Dict[str, Any]],
                           allowed_ids: Optional[Set[int]] = None) -> List[Dict[str, Any]]:
        """
        Generate common watchlist section.
        If allowed_ids is provided, ONLY include matching movies (e.g. prompt compliance).
        """
        # Get movies in multiple watchlists
        wl_subset = self.watchlist_df[
            self.watchlist_df['userId'].isin(group_users)
        ]
        
        # Count occurrences
        movie_counts = wl_subset.groupby('movieId')['userId'].apply(list).to_dict()
        
        # Filter: must appear in at least 2 users' watchlists
        common_watchlist = {
            mid: users for mid, users in movie_counts.items()
            if len(users) >= 2
        }
        
        # Exclude watched movies
        common_watchlist = {
            mid: users for mid, users in common_watchlist.items()
            if mid not in watched_set
        }
        
        if allowed_ids is not None:
            # Apply global filter
            common_watchlist = {
                mid: users for mid, users in common_watchlist.items()
                if mid in allowed_ids
            }

        # Exclude movies already in Section A
        section_a_ids = {rec['movie_id'] for rec in section_a}
        common_watchlist = {
            mid: users for mid, users in common_watchlist.items()
            if mid not in section_a_ids
        }
        
        # Build output
        section_b_output = []
        for movie_id, users in common_watchlist.items():
            title = self._get_movie_title(movie_id)
            users_sorted = sorted(users)
            explanation = f"Explicitly requested by member(s) {', '.join(map(str, users_sorted))}."
            
            # Get full metadata
            metadata = self._get_movie_metadata(movie_id)
            
            section_b_output.append({
                'movie_id': movie_id,
                'title': title,
                'users': users_sorted,
                'explanation': explanation,
                'signal_source': 'WATCHLIST',
                # Metadata fields
                'genres': metadata.get('genres'),
                'Overview': metadata.get('Overview'),
                'overview': metadata.get('overview'),
                'Director': metadata.get('Director'),
                'director': metadata.get('director'),
                'Actors': metadata.get('Actors'),
                'actors': metadata.get('actors'),
                'Production_Countries': metadata.get('Production_Countries'),
                'production_countries': metadata.get('production_countries'),
                'release_date': metadata.get('release_date'),
                'poster_url': metadata.get('poster_url'),
                'backdrop_url': metadata.get('backdrop_url'),
                'trailer_url': metadata.get('trailer_url')
            })
        
        # Sort by number of users (descending)
        section_b_output.sort(key=lambda x: len(x['users']), reverse=True)
        return section_b_output
    
    # ========================================================================
    # SECTION C: SHARED GROUP INTEREST THEMES
    # ========================================================================
    
    def _generate_section_c(self, group_users: List[int],
                           watched_set: Set[int],
                           section_a: List[Dict[str, Any]],
                           section_b: List[Dict[str, Any]],
                           allowed_ids: Optional[Set[int]] = None) -> List[Dict[str, Any]]:
        """
        Generate shared interest themes section with ENHANCED diversity.
        If allowed_ids is provided, ensure all recommended items match the global filter.
        """
        # Collect already recommended movie IDs
        excluded_ids = watched_set.copy()
        section_a_movie_ids = {rec['movie_id'] for rec in section_a}
        excluded_ids.update(section_a_movie_ids)
        excluded_ids.update({rec['movie_id'] for rec in section_b})
        
        # Extract themes
        themes = self._detect_diverse_themes(section_a, group_users)
        
        # Select top themes
        selected_themes = self._select_top_themes(themes, min_themes=5, max_themes=8)
        
        if not selected_themes:
            return []
        
        # Generate theme blocks
        section_c_output = []
        for theme in selected_themes:
            theme_block = self._generate_theme_block_enhanced(
                theme, group_users, excluded_ids, allowed_ids=allowed_ids
            )
            
            if theme_block:
                section_c_output.append(theme_block)
                excluded_ids.update({
                    m['movie_id'] for m in theme_block['recommended_movies']
                })
        
        return section_c_output
    
    def _detect_diverse_themes(self, section_a: List[Dict[str, Any]], 
                               group_users: List[int]) -> List[Dict[str, Any]]:
        """
        Detect themes from Section A movies using 6 different methods.
        """
        themes = []
        section_a_movie_ids = [rec['movie_id'] for rec in section_a]
        section_a_movies = self.movies_df[self.movies_df['movieId'].isin(section_a_movie_ids)]
        
        user_to_movies = {}
        for rec in section_a:
            for uid in rec.get('user_explanations', {}).keys():
                if uid not in user_to_movies:
                    user_to_movies[uid] = []
                user_to_movies[uid].append(rec['movie_id'])
        
        themes.extend(self._detect_director_themes(section_a_movies, user_to_movies))
        themes.extend(self._detect_actor_themes(section_a_movies, user_to_movies))
        themes.extend(self._detect_keyword_themes(section_a_movies, user_to_movies))
        themes.extend(self._detect_country_themes(section_a_movies, user_to_movies))
        themes.extend(self._detect_genre_themes(section_a_movies, user_to_movies))
        themes.extend(self._detect_year_themes(section_a_movies, user_to_movies))
        
        return themes

    # --- Theme Detection Helpers ---

    def _detect_director_themes(self, movies_df, user_to_movies):
        themes = []
        if 'Director' not in movies_df.columns: return themes
        
        director_to_movies = {}
        for _, row in movies_df.iterrows():
            if pd.notna(row.get('Director')):
                directors = [d.strip() for d in str(row['Director']).split(',')]
                for director in directors:
                    if director not in director_to_movies: director_to_movies[director] = []
                    director_to_movies[director].append(row['movieId'])
        
        for director, movie_ids in director_to_movies.items():
            if len(movie_ids) >= 2:
                user_movies = self._map_users_to_theme_movies(user_to_movies, movie_ids)
                if len(user_movies) >= 2:
                    themes.append({'theme_type': 'director', 'theme_value': director, 'user_movies': user_movies, 'score': len(user_movies)})
        return themes

    def _detect_actor_themes(self, movies_df, user_to_movies):
        themes = []
        if 'Actors' not in movies_df.columns: return themes
        
        actor_to_movies = {}
        for _, row in movies_df.iterrows():
            if pd.notna(row.get('Actors')):
                actors = [a.strip() for a in str(row['Actors']).split(',')]
                for actor in actors[:3]:
                    if actor not in actor_to_movies: actor_to_movies[actor] = []
                    actor_to_movies[actor].append(row['movieId'])

        for actor, movie_ids in actor_to_movies.items():
            if len(movie_ids) >= 2:
                user_movies = self._map_users_to_theme_movies(user_to_movies, movie_ids)
                if len(user_movies) >= 2:
                    themes.append({'theme_type': 'actor', 'theme_value': actor, 'user_movies': user_movies, 'score': len(user_movies)})
        return themes

    def _detect_keyword_themes(self, movies_df, user_to_movies):
        themes = []
        if 'Keywords' not in movies_df.columns: return themes
        
        keyword_to_movies = {}
        for _, row in movies_df.iterrows():
            if pd.notna(row.get('Keywords')):
                keywords = [k.strip() for k in str(row['Keywords']).split(',')]
                for keyword in keywords:
                    if keyword not in keyword_to_movies: keyword_to_movies[keyword] = []
                    keyword_to_movies[keyword].append(row['movieId'])

        for keyword, movie_ids in keyword_to_movies.items():
            if len(movie_ids) >= 2:
                user_movies = self._map_users_to_theme_movies(user_to_movies, movie_ids)
                if len(user_movies) >= 2:
                    themes.append({'theme_type': 'keyword', 'theme_value': keyword, 'user_movies': user_movies, 'score': len(user_movies)})
        return themes
        
    def _detect_country_themes(self, movies_df, user_to_movies):
        themes = []
        if 'Production_Countries' not in movies_df.columns: return themes
        
        country_to_movies = {}
        for _, row in movies_df.iterrows():
            if pd.notna(row.get('Production_Countries')):
                countries = [c.strip() for c in str(row['Production_Countries']).split(',')]
                for country in countries:
                    if country not in country_to_movies: country_to_movies[country] = []
                    country_to_movies[country].append(row['movieId'])

        for country, movie_ids in country_to_movies.items():
            if len(movie_ids) >= 2 and country != 'United States of America':
                user_movies = self._map_users_to_theme_movies(user_to_movies, movie_ids)
                if len(user_movies) >= 2:
                    themes.append({'theme_type': 'country', 'theme_value': country, 'user_movies': user_movies, 'score': len(user_movies)})
        return themes

    def _detect_genre_themes(self, movies_df, user_to_movies):
        themes = []
        if 'genres' not in movies_df.columns: return themes
        
        genre_to_movies = {}
        for _, row in movies_df.iterrows():
            if pd.notna(row.get('genres')):
                genres = str(row['genres']).split('|')
                if len(genres) >= 2:
                    combo = f"{genres[0]}-{genres[1]}" # Simple dual-genre
                    if combo not in genre_to_movies: genre_to_movies[combo] = []
                    genre_to_movies[combo].append(row['movieId'])

        for genre, movie_ids in genre_to_movies.items():
            if len(movie_ids) >= 2:
                user_movies = self._map_users_to_theme_movies(user_to_movies, movie_ids)
                if len(user_movies) >= 2:
                    themes.append({'theme_type': 'genre', 'theme_value': genre, 'user_movies': user_movies, 'score': len(user_movies)})
        return themes
    
    def _detect_year_themes(self, movies_df, user_to_movies):
        themes = []
        if 'release_date' not in movies_df.columns: return themes
        
        year_to_movies = {}
        for _, row in movies_df.iterrows():
            if pd.notna(row.get('release_date')):
                try:
                    year = int(str(row['release_date'])[:4])
                    decade = (year // 10) * 10
                    decade_str = f"{decade}s"
                    if decade_str not in year_to_movies: year_to_movies[decade_str] = []
                    year_to_movies[decade_str].append(row['movieId'])
                except: pass

        for decade, movie_ids in year_to_movies.items():
            if len(movie_ids) >= 2:
                user_movies = self._map_users_to_theme_movies(user_to_movies, movie_ids)
                if len(user_movies) >= 2:
                    themes.append({'theme_type': 'year', 'theme_value': decade, 'user_movies': user_movies, 'score': len(user_movies)})
        return themes

    def _map_users_to_theme_movies(self, user_to_movies, theme_movie_ids):
        user_movies = {}
        theme_set = set(theme_movie_ids)
        for uid, umovies in user_to_movies.items():
            common = set(umovies) & theme_set
            if common: user_movies[uid] = list(common)
        return user_movies

    def _select_top_themes(self, themes, min_themes=5, max_themes=8):
        # Sort by score (user count) then random shuffle for diversity
        themes.sort(key=lambda x: x['score'], reverse=True)
        # Unique check
        seen = set()
        unique = []
        for t in themes:
            k = f"{t['theme_type']}:{t['theme_value']}"
            if k not in seen:
                seen.add(k)
                unique.append(t)
        return unique[:max_themes]

    def _generate_theme_block_enhanced(self, theme, group_users, excluded_ids, allowed_ids: Optional[Set[int]] = None):
        # Find 3 candidate movies for this theme that are NOT in excluded
        # AND are in allowed_ids (if set)
        candidates = []
        theme_val = theme['theme_value']
        theme_type = theme['theme_type']
        
        # Simple search in movies_df for matching theme items
        # NOTE: This is naive search for candidates to fill the theme block
        # In production, use inverted index or richer search
        matches = pd.DataFrame()
        if theme_type == 'director':
            matches = self.movies_df[self.movies_df['Director'].str.contains(theme_val, na=False, regex=False)]
        elif theme_type == 'actor':
            matches = self.movies_df[self.movies_df['Actors'].str.contains(theme_val, na=False, regex=False)]
        elif theme_type == 'keyword':
            matches = self.movies_df[self.movies_df['Keywords'].str.contains(theme_val, na=False, regex=False)]
        elif theme_type == 'country':
            matches = self.movies_df[self.movies_df['Production_Countries'].str.contains(theme_val, na=False, regex=False)]
        
        # Filter popular ones
        if 'vote_count' in matches.columns and 'vote_average' in matches.columns:
            matches = matches[matches['vote_count'] > 50].sort_values('vote_average', ascending=False)
        elif 'vote_average' in matches.columns:
            matches = matches.sort_values('vote_average', ascending=False)
        
        rec_list = []
        for _, row in matches.iterrows():
            mid = row['movieId']
            
            # Global Filter Check
            if allowed_ids is not None:
                if mid not in allowed_ids:
                    continue
                    
            if mid not in excluded_ids:
                # Get full metadata
                metadata = self._get_movie_metadata(mid)
                
                rec_list.append({
                    'movie_id': mid,
                    'title': row['title'],
                    'poster_path': row.get('poster_path', ''),
                    # Metadata fields
                    'genres': metadata.get('genres'),
                    'Overview': metadata.get('Overview'),
                    'overview': metadata.get('overview'),
                    'Director': metadata.get('Director'),
                    'director': metadata.get('director'),
                    'Actors': metadata.get('Actors'),
                    'actors': metadata.get('actors'),
                    'Production_Countries': metadata.get('Production_Countries'),
                    'production_countries': metadata.get('production_countries'),
                    'release_date': metadata.get('release_date'),
                    'poster_url': metadata.get('poster_url'),
                    'backdrop_url': metadata.get('backdrop_url'),
                    'trailer_url': metadata.get('trailer_url')
                })
                if len(rec_list) >= 3: break
        
        if not rec_list: return None

        return {
            'theme_name': f"{theme_val} ({theme_type.title()})", # Renamed from theme_title
            'theme_type': theme_type,
            'justification': f"Shared interest based on movies liked by {', '.join(map(str, theme['user_movies'].keys()))}", # Renamed from explanation_basis
            'user_count': theme['score'], # Added
            'recommended_movies': rec_list
        }

    # --- Helpers ---
    
    def _get_watched_set(self, group_users: List[int]) -> Set[int]:
        watched = set()
        for uid in group_users:
            if uid in self.cf_matrix.index:
                user_ratings = self.cf_matrix.loc[uid].dropna()
                watched.update(user_ratings.index.tolist())
                print(f"[WATCHED] User {uid}: {len(user_ratings)} movies watched")
            else:
                print(f"[WATCHED] User {uid}: NOT FOUND in cf_matrix")
        print(f"[WATCHED] Total watched movies in group: {len(watched)}")
        return watched
    
    def _get_direct_watchlist_set(self, group_users: List[int]) -> Set[int]:
        """
        Returns only SHARED watchlist items (movies in 2+ users' watchlists).
        Individual watchlist items (1 user only) are NOT excluded from Top 10,
        as they represent valid recommendations for other group members.
        """
        wl_subset = self.watchlist_df[
            self.watchlist_df['userId'].isin(group_users)
        ]
        
        # Count how many users have each movie in their watchlist
        movie_user_counts = wl_subset.groupby('movieId')['userId'].nunique()
        
        # Only return movies that appear in 2+ users' watchlists
        shared_watchlist = set(movie_user_counts[movie_user_counts >= 2].index)
        
        print(f"[WATCHLIST] Total watchlist items: {len(wl_subset['movieId'].unique())}")
        print(f"[WATCHLIST] Shared items (2+ users): {len(shared_watchlist)}")
        
        return shared_watchlist
        
    def _get_movie_title(self, movie_id: int) -> str:
        if movie_id in self.movies_df['movieId'].values:
            return self.movies_df[
                self.movies_df['movieId'] == movie_id
            ]['title'].values[0]
        return f"Unknown Movie ({movie_id})"
    
    def _get_movie_metadata(self, movie_id: int) -> Dict[str, Any]:
        """
        Get full metadata for a movie from movies_df.
        Returns all available fields including TMDB data.
        """
        if movie_id not in self.movies_df['movieId'].values:
            return {}
        
        row = self.movies_df[self.movies_df['movieId'] == movie_id].iloc[0]
        
        
        # Convert row to dict and handle NaN values
        metadata = {}
        for col in row.index:
            val = row[col]
            if pd.notna(val):
                metadata[col] = val
            else:
                metadata[col] = None
        
        return metadata
    
    # ========================================================================
    # SECTION D: WATCHLIST-INSPIRED RECOMMENDATIONS
    # ========================================================================
    
    def _generate_section_d(self, group_users: List[int],
                           watched_set: Set[int],
                           section_a: List[Dict[str, Any]],
                           section_b: List[Dict[str, Any]],
                           allowed_ids: Optional[Set[int]] = None) -> List[Dict[str, Any]]:
        """
        Generate watchlist-inspired recommendations using Hybrid Model 3.
        These are NEW movies similar to watchlist items (not the watchlist items themselves).
        """
        # Get all watchlist movies for the group
        wl_subset = self.watchlist_df[
            self.watchlist_df['userId'].isin(group_users)
        ]
        
        watchlist_movies = set(wl_subset['movieId'].unique())
        
        # Exclude watched and already recommended movies
        excluded_ids = watched_set.copy()
        excluded_ids.update({rec['movie_id'] for rec in section_a})
        excluded_ids.update({item['movie_id'] for item in section_b})
        excluded_ids.update(watchlist_movies)  # Don't recommend watchlist items themselves
        
        # Get candidates similar to watchlist items
        # Use content-based similarity from H3's cb_model
        candidates = set()
        
        for wl_movie in watchlist_movies:
            # Get similar movies using content-based model
            if wl_movie in self.h3.cb_model.movie_to_idx:
                wl_idx = self.h3.cb_model.movie_to_idx[wl_movie]
                wl_vec = self.h3.cb_model.tfidf_matrix[wl_idx]
                
                # Calculate similarity with all movies
                all_sims = cosine_similarity(wl_vec, self.h3.cb_model.tfidf_matrix).flatten()
                
                # Get top 20 similar movies
                top_indices = np.argsort(all_sims)[-21:-1][::-1]  # Exclude self
                
                # Create reverse mapping for this iteration
                idx_to_movie = {v: k for k, v in self.h3.cb_model.movie_to_idx.items()}
                
                for idx in top_indices:
                    if idx in idx_to_movie:
                        similar_movie_id = idx_to_movie[idx]
                        if similar_movie_id not in excluded_ids:
                            candidates.add(similar_movie_id)
        
        if not candidates:
            return []
        
        # Use Hybrid Model 3 to rank these candidates
        try:
            recommendations = self.h3.recommend_for_group(
                group_users, list(candidates), top_k=10
            )
        except Exception as e:
            print(f"[SECTION D] H3 recommendation failed: {e}")
            return []
        
        # Apply global filter if provided
        if allowed_ids is not None:
            recommendations = [
                rec for rec in recommendations 
                if rec['movie_id'] in allowed_ids
            ]
        
        # Build output
        section_d_output = []
        for rec in recommendations[:10]:
            movie_id = rec['movie_id']
            title = self._get_movie_title(movie_id)
            
            # Get metadata
            metadata = self._get_movie_metadata(movie_id)
            
            # Create explanation
            explanation = f"Recommended based on your group's watchlist preferences."
            
            section_d_output.append({
                'movie_id': movie_id,
                'title': title,
                'group_score': round(rec['score'], 2),
                'explanation': explanation,
                'signal_source': 'WATCHLIST_INSPIRED',
                'source_model': 'H3',
                # Metadata fields
                'genres': metadata.get('genres'),
                'Overview': metadata.get('Overview'),
                'overview': metadata.get('overview'),
                'Director': metadata.get('Director'),
                'director': metadata.get('director'),
                'Actors': metadata.get('Actors'),
                'actors': metadata.get('actors'),
                'Production_Countries': metadata.get('Production_Countries'),
                'production_countries': metadata.get('production_countries'),
                'release_date': metadata.get('release_date'),
                'poster_url': metadata.get('poster_url'),
                'backdrop_url': metadata.get('backdrop_url'),
                'trailer_url': metadata.get('trailer_url')
            })
        
        return section_d_output
    
    # ========================================================================
    # SECTION E: HYBRID 1 SPECIFIC RECOMMENDATIONS
    # ========================================================================
    
    def _generate_section_e(self, group_users: List[int],
                           watched_set: Set[int],
                           section_a: List[Dict[str, Any]],
                           section_b: List[Dict[str, Any]],
                           section_d: List[Dict[str, Any]],
                           allowed_ids: Optional[Set[int]] = None) -> List[Dict[str, Any]]:
        """
        Generate Hybrid Model 1 specific recommendations.
        Focus on collaborative filtering and similar taste patterns.
        """
        # Exclude already recommended movies
        excluded_ids = watched_set.copy()
        excluded_ids.update({rec['movie_id'] for rec in section_a})
        excluded_ids.update({item['movie_id'] for item in section_b})
        excluded_ids.update({item['movie_id'] for item in section_d})
        
        # Get candidates
        candidates = self._get_candidates_for_section_a(
            group_users, watched_set, set()
        )
        
        # Filter out excluded
        candidates = [c for c in candidates if c not in excluded_ids]
        
        if not candidates:
            return []
        
        # Use Hybrid Model 1
        try:
            recommendations = self.h1.recommend_for_group(
                group_users, candidates, top_k=10
            )
        except Exception as e:
            print(f"[SECTION E] H1 recommendation failed: {e}")
            return []
        
        # Apply global filter if provided
        if allowed_ids is not None:
            recommendations = [
                rec for rec in recommendations 
                if rec['movie_id'] in allowed_ids
            ]
        
        # Build output
        section_e_output = []
        for rec in recommendations[:10]:
            movie_id = rec['movie_id']
            title = self._get_movie_title(movie_id)
            metadata = self._get_movie_metadata(movie_id)
            
            section_e_output.append({
                'movie_id': movie_id,
                'title': title,
                'group_score': round(rec['score'], 2),
                'explanation': "Because you have similar tastes.",
                'signal_source': 'COLLABORATIVE',
                'source_model': 'H1',
                'genres': metadata.get('genres'),
                'Overview': metadata.get('Overview'),
                'overview': metadata.get('overview'),
                'Director': metadata.get('Director'),
                'director': metadata.get('director'),
                'Actors': metadata.get('Actors'),
                'actors': metadata.get('actors'),
                'Production_Countries': metadata.get('Production_Countries'),
                'production_countries': metadata.get('production_countries'),
                'release_date': metadata.get('release_date'),
                'poster_url': metadata.get('poster_url'),
                'backdrop_url': metadata.get('backdrop_url'),
                'trailer_url': metadata.get('trailer_url')
            })
        
        return section_e_output
    
    # ========================================================================
    # SECTION F: HYBRID 2 SPECIFIC RECOMMENDATIONS
    # ========================================================================
    
    def _generate_section_f(self, group_users: List[int],
                           watched_set: Set[int],
                           section_a: List[Dict[str, Any]],
                           section_b: List[Dict[str, Any]],
                           section_d: List[Dict[str, Any]],
                           allowed_ids: Optional[Set[int]] = None) -> List[Dict[str, Any]]:
        """
        Generate Hybrid Model 2 specific recommendations.
        Adaptive strategy switching based on user profiles.
        """
        # Exclude already recommended movies
        excluded_ids = watched_set.copy()
        excluded_ids.update({rec['movie_id'] for rec in section_a})
        excluded_ids.update({item['movie_id'] for item in section_b})
        excluded_ids.update({item['movie_id'] for item in section_d})
        
        # Get candidates
        candidates = self._get_candidates_for_section_a(
            group_users, watched_set, set()
        )
        
        # Filter out excluded
        candidates = [c for c in candidates if c not in excluded_ids]
        
        if not candidates:
            return []
        
        # Use Hybrid Model 2
        try:
            recommendations = self.h2.recommend_for_group(
                group_users, candidates, top_k=10
            )
        except Exception as e:
            print(f"[SECTION F] H2 recommendation failed: {e}")
            return []
        
        # Apply global filter if provided
        if allowed_ids is not None:
            recommendations = [
                rec for rec in recommendations 
                if rec['movie_id'] in allowed_ids
            ]
        
        # Build output
        section_f_output = []
        for rec in recommendations[:10]:
            movie_id = rec['movie_id']
            title = self._get_movie_title(movie_id)
            metadata = self._get_movie_metadata(movie_id)
            
            section_f_output.append({
                'movie_id': movie_id,
                'title': title,
                'group_score': round(rec['score'], 2),
                'explanation': "Adaptive recommendation based on your profiles.",
                'signal_source': 'ADAPTIVE',
                'source_model': 'H2',
                'genres': metadata.get('genres'),
                'Overview': metadata.get('Overview'),
                'overview': metadata.get('overview'),
                'Director': metadata.get('Director'),
                'director': metadata.get('director'),
                'Actors': metadata.get('Actors'),
                'actors': metadata.get('actors'),
                'Production_Countries': metadata.get('Production_Countries'),
                'production_countries': metadata.get('production_countries'),
                'release_date': metadata.get('release_date'),
                'poster_url': metadata.get('poster_url'),
                'backdrop_url': metadata.get('backdrop_url'),
                'trailer_url': metadata.get('trailer_url')
            })
        
        return section_f_output
