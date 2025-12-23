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
        
        # Generate Section A (multi-model or single model)
        if self.use_multi_model_selection and self.model_performance:
            section_a = self._generate_section_a_multi_model(
                group_users, watched_set, direct_watchlist_set, user_prompt=user_prompt
            )
        else:
            section_a = self._generate_section_a(
                group_users, watched_set, direct_watchlist_set, user_prompt=user_prompt
            )
        
        # Generate Section B
        section_b = self._generate_section_b(
            group_users, watched_set, section_a
        )
        
        # Generate Section C
        section_c = self._generate_section_c(
            group_users, watched_set, section_a, section_b
        )
        
        return {
            'section_a_top_recommendations': section_a,
            'section_b_common_watchlist': section_b,
            'section_c_shared_interests': section_c
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
    
    def _get_candidates_for_agent(self, group_users: List[int], 
                                 watched_set: Set[int], 
                                 direct_watchlist_set: Set[int], 
                                 top_k: int = 50) -> List[Dict[str, Any]]:
        """
        Generate a rich list of candidates for the AI Agent to process.
        Returns a list of Dicts with full metadata (overview, cast, etc.).
        """
        # Get raw candidate IDs
        candidate_ids = self._get_candidates_for_section_a(group_users, watched_set, direct_watchlist_set)
        
        # Take top-K candidates (scored by Hybrid Model 1 as a baseline)
        scored_recs = self.h1.recommend_for_group(group_users, candidate_ids, top_k=top_k)
        
        # Enrich with metadata
        enriched_candidates = []
        for rec in scored_recs:
            movie_id = rec['movie_id']
            # Lookup movie details
            movie_row = self.movies_df[self.movies_df['movieId'] == movie_id]
            if not movie_row.empty:
                row = movie_row.iloc[0]
                enriched_candidates.append({
                    'movieId': int(movie_id),
                    'title': str(row.get('title', 'Unknown')),
                    'genres': str(row.get('genres', '')).split('|'),
                    'overview': str(row.get('overview', '')),
                    'keywords': str(row.get('keywords', '')),
                    'actors': str(row.get('actors', '')),
                    'director': str(row.get('director', '')),
                    'runtime': int(row.get('runtime', 0)) if pd.notna(row.get('runtime')) else 0,
                    'vote_average': float(row.get('vote_average', 0.0)) if pd.notna(row.get('vote_average')) else 0.0,
                    'baseline_score': rec['score'],
                    'explanations': rec['explanations'] # Keep original explanations
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
        user_prompt: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate Top-10 using multi-model selection strategy.
        Supports Agent Filtering if user_prompt is provided.
        """
        
        # --- AGENT PATH ---
        if user_prompt and HAS_AGENT:
            # 1. Get rich candidates
            rich_candidates = self._get_candidates_for_agent(
                group_users, watched_set, direct_watchlist_set, top_k=60
            ) 
            
            # 2. Initialize temporary agent if not passed in init
            agent_to_use = self.film_agent
            if not agent_to_use:
                try:
                    agent_to_use = get_filter_agent()
                except Exception as e:
                    print(f"[ERROR] Could not init agent on fly: {e}")
            
            # 3. Apply Agent Filter
            if agent_to_use:
                # We need to temporarily set self.film_agent if it was None
                original_agent = self.film_agent
                self.film_agent = agent_to_use
                
                filtered_candidates = self._apply_agent_filter(user_prompt, rich_candidates)
                
                # Restore
                self.film_agent = original_agent
                
                if filtered_candidates:
                    print(f"[AGENT] Success. Returning {len(filtered_candidates)} filtered items.")
                    # Format for Section A output
                    output = []
                    for c in filtered_candidates[:10]:
                        output.append({
                            'movie_id': c['movieId'],
                            'title': c['title'],
                            'group_score': round(c['baseline_score'], 2),
                            'source_model': 'AI_AGENT', # Mark source
                            'group_explanation': f"AI Agent Match: {c.get('agent_reasoning', 'Matches your request.')}",
                            'signal_source': 'AI_AGENT',
                            'user_explanations': c.get('explanations', {})
                        })
                    return output
            
            print("[AGENT] Fallback to standard logic (agent failed or returned 0).")
        
        # --- STANDARD PATH ---
        
        print("\n[SECTION A] Using MULTI-MODEL selection strategy (Standard)")
        
        # Calculate quota allocation
        quota = self._calculate_quota_allocation(total_slots=10)
        
        # Generate candidates from each model
        model_candidates = {}
        
        for model_key in ['h1', 'h2', 'h3']:
            model = getattr(self, model_key)
            
            # Get candidates (same logic as original)
            candidates = self._get_candidates_for_section_a(
                group_users, watched_set, direct_watchlist_set
            )
            
            if not candidates:
                model_candidates[model_key] = []
                continue
            
            # Request more than quota to allow filtering
            request_k = quota[model_key] * 3
            
            try:
                recs = model.recommend_for_group(
                    group_users, candidates, top_k=request_k
                )
                
                # Filter out direct watchlist matches
                filtered_recs = []
                for rec in recs:
                    if rec['movie_id'] not in direct_watchlist_set:
                        # Add source model attribution
                        rec['source_model'] = model_key.upper()
                        filtered_recs.append(rec)
                
                model_candidates[model_key] = filtered_recs
                
            except Exception as e:
                print(f"[WARNING] {model_key.upper()} failed: {e}")
                model_candidates[model_key] = []
        
        # Select top films from each model according to quota
        final_recommendations = []
        used_movie_ids = set()
        
        for model_key in ['h1', 'h2', 'h3']:
            quota_for_model = quota[model_key]
            candidates = model_candidates[model_key]
            
            selected_count = 0
            for rec in candidates:
                if selected_count >= quota_for_model:
                    break
                
                # Skip if already selected by another model
                if rec['movie_id'] in used_movie_ids:
                    continue
                
                # Add to final list
                final_recommendations.append(rec)
                used_movie_ids.add(rec['movie_id'])
                selected_count += 1
        
        # Sort by score (descending)
        final_recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        # Format output
        section_a_output = []
        for rec in final_recommendations[:10]:
            movie_id = rec['movie_id']
            title = self._get_movie_title(movie_id)
            
            # Filter explanations (no direct watchlist)
            user_explanations = {}
            for uid, expl in rec['explanations'].items():
                if not self._is_direct_watchlist_explanation(expl):
                    user_explanations[uid] = expl
            
            if not user_explanations:
                continue
            
            section_a_output.append({
                'movie_id': movie_id,
                'title': title,
                'group_score': round(rec['score'], 2),
                'source_model': rec['source_model'],
                'model_score': round(rec['score'], 2),
                'group_explanation': rec['group_explanation'],
                'signal_source': self._get_dominant_signal_source(user_explanations),
                'user_explanations': user_explanations
            })
        
        return section_a_output[:10]
    
    def _generate_section_a(self, group_users: List[int], 
                           watched_set: Set[int],
                           direct_watchlist_set: Set[int],
                           user_prompt: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Generate Top-10 ranked group recommendations (Legacy Single Model).
        Also supports Agent Filtering now.
        """
        
        # --- AGENT PATH ---
        if user_prompt and HAS_AGENT:
             # Reuse logic from multi-model function for agent (it's model agnostic)
            rich_candidates = self._get_candidates_for_agent(
                group_users, watched_set, direct_watchlist_set, top_k=60
            )
            
            # Initialize temporary agent if needed
            agent_to_use = self.film_agent
            if not agent_to_use:
                try:
                    agent_to_use = get_filter_agent()
                except Exception:
                    pass

            if agent_to_use:
                original_agent = self.film_agent
                self.film_agent = agent_to_use
                filtered_candidates = self._apply_agent_filter(user_prompt, rich_candidates)
                self.film_agent = original_agent
                
                if filtered_candidates:
                    output = []
                    for c in filtered_candidates[:10]:
                        output.append({
                            'movie_id': c['movieId'],
                            'title': c['title'],
                            'group_score': round(c['baseline_score'], 2),
                            'source_model': 'AI_AGENT',
                            'group_explanation': f"AI Agent Match: {c.get('agent_reasoning', 'Matches request')}",
                            'signal_source': 'AI_AGENT',
                            'user_explanations': c.get('original_explanations', c.get('explanations', {}))
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
            for uid, expl in rec['explanations'].items():
                if self._is_direct_watchlist_explanation(expl):
                    continue
                user_explanations[uid] = expl
            
            if not user_explanations:
                continue
            
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
                if hasattr(self.h1.ib_model, 'neighbors'):
                    neighbors_dict = self.h1.ib_model.neighbors.get(wl_movie, {})
                    similar_movies = list(neighbors_dict.keys())[:5]
                    candidates.update([m for m in similar_movies if m != wl_movie])
        
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
                           section_a: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate common watchlist section.
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
            
            section_b_output.append({
                'movie_id': movie_id,
                'title': title,
                'users': users_sorted,
                'explanation': explanation,
                'signal_source': 'WATCHLIST'
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
                           section_b: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate shared interest themes section with ENHANCED diversity.
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
                theme, group_users, excluded_ids
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

    def _generate_theme_block_enhanced(self, theme, group_users, excluded_ids):
        # Find 3 candidate movies for this theme that are NOT in excluded
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
            if mid not in excluded_ids:
                rec_list.append({
                    'movie_id': mid,
                    'title': row['title'],
                    'poster_path': row.get('poster_path', '') # Assuming column exists or frontend handles
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
        return watched
    
    def _get_direct_watchlist_set(self, group_users: List[int]) -> Set[int]:
        wl_subset = self.watchlist_df[
            self.watchlist_df['userId'].isin(group_users)
        ]
        return set(wl_subset['movieId'].unique())
        
    def _get_movie_title(self, movie_id: int) -> str:
        if movie_id in self.movies_df['movieId'].values:
            return self.movies_df[
                self.movies_df['movieId'] == movie_id
            ]['title'].values[0]
        return f"Unknown Movie ({movie_id})"
