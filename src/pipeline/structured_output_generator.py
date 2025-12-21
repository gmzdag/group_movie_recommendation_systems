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


class StructuredOutputGenerator:
    """
    Generates structured, text-based recommendation outputs.
    
    This class does NOT design UI, does NOT mention frontend/layout.
    It produces JSON-serializable data structures ready for UI consumption.
    """
    
    def __init__(self, hybrid_model_1, hybrid_model_2, hybrid_model_3, 
                 movies_df, watchlist_df, cf_matrix, ratings_df=None,
                 enable_temporal_filtering=True, use_multi_model_selection=True):
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
        """
        self.h1 = hybrid_model_1
        self.h2 = hybrid_model_2
        self.h3 = hybrid_model_3
        self.movies_df = movies_df
        self.watchlist_df = watchlist_df
        self.cf_matrix = cf_matrix
        self.enable_temporal_filtering = enable_temporal_filtering
        self.use_multi_model_selection = use_multi_model_selection
        
        # Initialize temporal analyzer if ratings data is provided
        if ratings_df is not None and enable_temporal_filtering:
            self.temporal_analyzer = TemporalPreferenceAnalyzer(ratings_df, movies_df)
        else:
            self.temporal_analyzer = None
        
        # Load model performance for multi-model selection
        self.model_performance = self._load_model_performance() if use_multi_model_selection else None
        
    def generate_three_section_output(self, group_users: List[int]) -> Dict[str, Any]:
        """
        Generates all three sections for a group.
        
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
                group_users, watched_set, direct_watchlist_set
            )
        else:
            section_a = self._generate_section_a(
                group_users, watched_set, direct_watchlist_set
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
                print(f"[INFO] Loaded model performance: "
                      f"H1=#{performance['h1']['rank']}, "
                      f"H2=#{performance['h2']['rank']}, "
                      f"H3=#{performance['h3']['rank']}")
                return performance
            except Exception as e:
                print(f"[WARNING] Could not load model performance: {e}")
                return None
        else:
            print(f"[WARNING] Model performance file not found at {cache_path}")
            print("[INFO] Run 'python src/calibration/train_production.py' to generate it.")
            return None
    
    def _calculate_quota_allocation(self, total_slots: int = 10) -> Dict[str, int]:
        """
        Calculate slot allocation for each model based on performance ranking.
        
        Strategy: Performance-proportional allocation
        - Rank 1 gets most slots
        - Rank 2 gets moderate slots  
        - Rank 3 gets fewest slots
        
        Example: For 10 slots with ranks [1,2,3] -> [5, 3, 2]
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
    
    def _generate_section_a_multi_model(
        self,
        group_users: List[int],
        watched_set: Set[int],
        direct_watchlist_set: Set[int]
    ) -> List[Dict[str, Any]]:
        """
        Generate Top-10 using multi-model selection strategy.
        
        Each film comes from a single model based on performance ranking.
        Models do NOT merge scores - each recommendation has a source model.
        
        STRICT RULES:
        - Watchlist direct matches EXCLUDED
        - Each film attributed to source model
        - Quota-based selection (performance-proportional)
        """
        print("\n[SECTION A] Using MULTI-MODEL selection strategy")
        
        # Calculate quota allocation
        quota = self._calculate_quota_allocation(total_slots=10)
        print(f"[QUOTA] H1={quota['h1']}, H2={quota['h2']}, H3={quota['h3']}")
        
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
                print(f"[{model_key.upper()}] Generated {len(filtered_recs)} candidates")
                
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
        
        print(f"[SECTION A] Final: {len(section_a_output)} recommendations")
        return section_a_output[:10]
    
    def _generate_section_a(self, group_users: List[int], 
                           watched_set: Set[int],
                           direct_watchlist_set: Set[int]) -> List[Dict[str, Any]]:
        """
        Generate Top-10 ranked group recommendations.
        
        STRICT RULES:
        - WATCHLIST DIRECT MATCH explanations are FORBIDDEN
        - Movies in direct watchlist MUST be EXCLUDED
        - WATCHLIST signals allowed ONLY if similarity-based
        - Use AVERAGE STRATEGY for aggregation
        - Apply sequel filtering
        """
        # 1. Generate candidates
        candidates = self._get_candidates_for_section_a(
            group_users, watched_set, direct_watchlist_set
        )
        
        if not candidates:
            return []
        
        # 2. Use Hybrid Model 1 for recommendations
        # (Dynamic Weighted Hybrid: IBCF + CBF)
        recommendations = self.h1.recommend_for_group(
            group_users, candidates, top_k=10
        )
        
        # 3. Post-process to ensure watchlist direct matches are excluded
        # and explanations are properly formatted
        section_a_output = []
        
        for rec in recommendations:
            movie_id = rec['movie_id']
            
            # Double-check: Exclude direct watchlist matches
            if movie_id in direct_watchlist_set:
                continue
            
            # Get movie title
            title = self._get_movie_title(movie_id)
            
            # Process user explanations to ensure no direct watchlist matches
            user_explanations = {}
            for uid, expl in rec['explanations'].items():
                # Filter out direct watchlist explanations
                if self._is_direct_watchlist_explanation(expl):
                    # Skip or replace with similarity-based if possible
                    # For Section A, we should not have direct matches
                    # This is a safety check
                    continue
                user_explanations[uid] = expl
            
            # If no valid explanations remain, skip this movie
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
        
        return section_a_output[:10]  # Ensure exactly top 10
    
    def _get_candidates_for_section_a(self, group_users: List[int],
                                      watched_set: Set[int],
                                      direct_watchlist_set: Set[int]) -> List[int]:
        """
        Generate candidate movies for Section A.
        
        Strategy:
        1. IBCF neighbors (similar to liked movies)
        2. UBCF neighbors (liked by similar users)
        3. Watchlist similarity (NOT direct matches)
        4. Popular baseline
        5. Temporal filtering (NEW: filter by release year preferences)
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
                # Get similar movies to watchlist items
                if hasattr(self.h1.ib_model, 'neighbors'):
                    neighbors_dict = self.h1.ib_model.neighbors.get(wl_movie, {})
                    similar_movies = list(neighbors_dict.keys())[:5]
                    # Add only similar movies, NOT the watchlist movie itself
                    candidates.update([m for m in similar_movies if m != wl_movie])
        
        # 3. Popular baseline
        popular = self.cf_matrix.count().sort_values(
            ascending=False
        ).head(200).index.tolist()
        candidates.update(popular)
        
        # 4. Filter exclusions (watched and direct watchlist)
        candidates = [
            c for c in candidates 
            if c not in watched_set and c not in direct_watchlist_set
        ]
        
        # 5. Apply temporal filtering (NEW)
        if self.temporal_analyzer is not None:
            # Get group temporal profile
            group_profile = self.temporal_analyzer.get_group_temporal_profile(group_users)
            
            # Filter candidates by temporal compatibility
            temporally_filtered = []
            for movie_id in candidates:
                # Check if movie is temporally compatible for the group
                # Use lenient mode (not strict) to allow some flexibility
                is_compatible = True
                for uid in group_users:
                    if not self.temporal_analyzer.is_movie_temporally_compatible(
                        uid, movie_id, strict=False
                    ):
                        is_compatible = False
                        break
                
                if is_compatible:
                    temporally_filtered.append(movie_id)
            
            # If temporal filtering is too restrictive, fall back to original candidates
            if len(temporally_filtered) < 20:
                print(f"[WARNING] Temporal filtering too restrictive "
                      f"({len(temporally_filtered)} candidates). Using original set.")
            else:
                candidates = temporally_filtered
                print(f"[INFO] Temporal filtering applied: {len(candidates)} candidates remain.")
        
        return candidates
    
    def _is_direct_watchlist_explanation(self, explanation: Dict[str, Any]) -> bool:
        """
        Check if explanation is a direct watchlist match.
        
        Direct match patterns:
        - "This movie is in your watchlist."
        - Contains "Direct Match" in features
        """
        primary = explanation.get('primary_reason', '')
        
        if 'This movie is in your watchlist' in primary:
            return True
        if 'Direct Match' in primary:
            return True
        
        return False
    
    def _get_dominant_signal_source(self, user_explanations: Dict[int, Dict]) -> str:
        """Extract dominant signal source from user explanations."""
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
        
        STRICT RULES:
        - NOT ranked
        - WATCHLIST DIRECT MATCH logic ONLY
        - Movies MUST NOT appear in Section A
        - Hybrid Models 1 and 2 MUST NOT be used
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
            
            # Sort users for deterministic output
            users_sorted = sorted(users)
            
            explanation = f"Explicitly requested by member(s) {', '.join(map(str, users_sorted))}."
            
            section_b_output.append({
                'movie_id': movie_id,
                'title': title,
                'users': users_sorted,
                'explanation': explanation,
                'signal_source': 'WATCHLIST'
            })
        
        # Sort by number of users (descending) for consistent ordering
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
        
        NEW APPROACH:
        - Detect themes from 6 types: Director, Actor, Keyword, Country, Genre, Year
        - Minimum 5 themes, maximum 8 themes
        - Each theme must cite specific users
        - Avoid generalizations
        
        STRICT RULES:
        - Derive interests from Section A movies (not raw metadata)
        - DO NOT use high-level genres alone
        - Each theme must have user attribution
        """
        # Collect already recommended movie IDs
        excluded_ids = watched_set.copy()
        section_a_movie_ids = {rec['movie_id'] for rec in section_a}
        excluded_ids.update(section_a_movie_ids)
        excluded_ids.update({rec['movie_id'] for rec in section_b})
        
        # Extract themes from Section A movies using 6 detection methods
        themes = self._detect_diverse_themes(section_a, group_users)
        
        # Select top themes (minimum 5, maximum 8)
        selected_themes = self._select_top_themes(themes, min_themes=5, max_themes=8)
        
        if not selected_themes:
            print("[SECTION C] No themes detected")
            return []
        
        print(f"[SECTION C] Selected {len(selected_themes)} diverse themes")
        
        # Generate theme blocks
        section_c_output = []
        
        for theme in selected_themes:
            theme_block = self._generate_theme_block_enhanced(
                theme, group_users, excluded_ids
            )
            
            if theme_block:
                section_c_output.append(theme_block)
                # Update excluded IDs to prevent duplicates
                excluded_ids.update({
                    m['movie_id'] for m in theme_block['recommended_movies']
                })
        
        return section_c_output
    
    def _detect_diverse_themes(self, section_a: List[Dict[str, Any]], 
                               group_users: List[int]) -> List[Dict[str, Any]]:
        """
        Detect themes from Section A movies using 6 different methods.
        
        Returns list of themes with:
        - theme_type: 'director', 'actor', 'keyword', 'country', 'genre', 'year'
        - theme_value: The actual value (e.g., 'Christopher Nolan')
        - user_movies: Dict mapping user_id to list of movie_ids supporting this theme
        - score: Number of users who have this theme
        """
        themes = []
        
        # Get Section A movie IDs
        section_a_movie_ids = [rec['movie_id'] for rec in section_a]
        section_a_movies = self.movies_df[self.movies_df['movieId'].isin(section_a_movie_ids)]
        
        # Build user-to-movies mapping from Section A
        user_to_movies = {}
        for rec in section_a:
            for uid in rec.get('user_explanations', {}).keys():
                if uid not in user_to_movies:
                    user_to_movies[uid] = []
                user_to_movies[uid].append(rec['movie_id'])
        
        # 1. DIRECTOR themes
        themes.extend(self._detect_director_themes(section_a_movies, user_to_movies))
        
        # 2. ACTOR themes
        themes.extend(self._detect_actor_themes(section_a_movies, user_to_movies))
        
        # 3. KEYWORD themes
        themes.extend(self._detect_keyword_themes(section_a_movies, user_to_movies))
        
        # 4. COUNTRY themes
        themes.extend(self._detect_country_themes(section_a_movies, user_to_movies))
        
        # 5. GENRE themes (specific, not high-level)
        themes.extend(self._detect_genre_themes(section_a_movies, user_to_movies))
        
        # 6. YEAR/ERA themes
        themes.extend(self._detect_year_themes(section_a_movies, user_to_movies))
        
        return themes
    
    def _detect_director_themes(self, movies_df, user_to_movies) -> List[Dict]:
        """Detect director themes from Section A movies"""
        themes = []
        
        if 'Director' not in movies_df.columns:
            return themes
        
        # Extract all directors
        director_to_movies = {}
        for _, row in movies_df.iterrows():
            if pd.notna(row.get('Director')):
                directors = [d.strip() for d in str(row['Director']).split(',')]
                for director in directors:
                    if director not in director_to_movies:
                        director_to_movies[director] = []
                    director_to_movies[director].append(row['movieId'])
        
        # Build themes
        for director, movie_ids in director_to_movies.items():
            if len(movie_ids) >= 2:  # At least 2 movies
                user_movies = {}
                for uid, umovies in user_to_movies.items():
                    common = set(umovies) & set(movie_ids)
                    if common:
                        user_movies[uid] = list(common)
                
                if len(user_movies) >= 2:  # At least 2 users
                    themes.append({
                        'theme_type': 'director',
                        'theme_value': director,
                        'user_movies': user_movies,
                        'score': len(user_movies)
                    })
        
        return themes
    
    def _detect_actor_themes(self, movies_df, user_to_movies) -> List[Dict]:
        """Detect actor themes from Section A movies"""
        themes = []
        
        if 'Actors' not in movies_df.columns:
            return themes
        
        # Extract all actors
        actor_to_movies = {}
        for _, row in movies_df.iterrows():
            if pd.notna(row.get('Actors')):
                actors = [a.strip() for a in str(row['Actors']).split(',')]
                for actor in actors[:3]:  # Top 3 actors only
                    if actor not in actor_to_movies:
                        actor_to_movies[actor] = []
                    actor_to_movies[actor].append(row['movieId'])
        
        # Build themes (only actors with 2+ movies)
        for actor, movie_ids in actor_to_movies.items():
            if len(movie_ids) >= 2:
                user_movies = {}
                for uid, umovies in user_to_movies.items():
                    common = set(umovies) & set(movie_ids)
                    if common:
                        user_movies[uid] = list(common)
                
                if len(user_movies) >= 2:
                    themes.append({
                        'theme_type': 'actor',
                        'theme_value': actor,
                        'user_movies': user_movies,
                        'score': len(user_movies)
                    })
        
        return themes
    
    def _detect_keyword_themes(self, movies_df, user_to_movies) -> List[Dict]:
        """Detect keyword themes from Section A movies"""
        themes = []
        
        if 'Keywords' not in movies_df.columns:
            return themes
        
        # Extract all keywords
        keyword_to_movies = {}
        for _, row in movies_df.iterrows():
            if pd.notna(row.get('Keywords')):
                keywords = [k.strip() for k in str(row['Keywords']).split(',')]
                for keyword in keywords:
                    if keyword not in keyword_to_movies:
                        keyword_to_movies[keyword] = []
                    keyword_to_movies[keyword].append(row['movieId'])
        
        # Build themes (only keywords with 2+ movies)
        for keyword, movie_ids in keyword_to_movies.items():
            if len(movie_ids) >= 2:
                user_movies = {}
                for uid, umovies in user_to_movies.items():
                    common = set(umovies) & set(movie_ids)
                    if common:
                        user_movies[uid] = list(common)
                
                if len(user_movies) >= 2:
                    themes.append({
                        'theme_type': 'keyword',
                        'theme_value': keyword,
                        'user_movies': user_movies,
                        'score': len(user_movies)
                    })
        
        return themes
    
    def _detect_country_themes(self, movies_df, user_to_movies) -> List[Dict]:
        """Detect country/region themes from Section A movies"""
        themes = []
        
        if 'Production_Countries' not in movies_df.columns:
            return themes
        
        # Extract all countries
        country_to_movies = {}
        for _, row in movies_df.iterrows():
            if pd.notna(row.get('Production_Countries')):
                countries = [c.strip() for c in str(row['Production_Countries']).split(',')]
                for country in countries:
                    if country not in country_to_movies:
                        country_to_movies[country] = []
                    country_to_movies[country].append(row['movieId'])
        
        # Build themes (only countries with 2+ movies, exclude USA if too common)
        for country, movie_ids in country_to_movies.items():
            if len(movie_ids) >= 2 and country != 'United States of America':
                user_movies = {}
                for uid, umovies in user_to_movies.items():
                    common = set(umovies) & set(movie_ids)
                    if common:
                        user_movies[uid] = list(common)
                
                if len(user_movies) >= 2:
                    themes.append({
                        'theme_type': 'country',
                        'theme_value': country,
                        'user_movies': user_movies,
                        'score': len(user_movies)
                    })
        
        return themes
    
    def _detect_genre_themes(self, movies_df, user_to_movies) -> List[Dict]:
        """Detect specific genre combinations (not high-level)"""
        themes = []
        
        if 'genres' not in movies_df.columns:
            return themes
        
        # Extract genre combinations
        genre_to_movies = {}
        for _, row in movies_df.iterrows():
            if pd.notna(row.get('genres')):
                genres = str(row['genres'])
                # Only use specific combinations, not single genres
                if '|' in genres:  # Multi-genre
                    if genres not in genre_to_movies:
                        genre_to_movies[genres] = []
                    genre_to_movies[genres].append(row['movieId'])
        
        # Build themes
        for genre_combo, movie_ids in genre_to_movies.items():
            if len(movie_ids) >= 2:
                user_movies = {}
                for uid, umovies in user_to_movies.items():
                    common = set(umovies) & set(movie_ids)
                    if common:
                        user_movies[uid] = list(common)
                
                if len(user_movies) >= 2:
                    themes.append({
                        'theme_type': 'genre',
                        'theme_value': genre_combo.replace('|', ' + '),
                        'user_movies': user_movies,
                        'score': len(user_movies)
                    })
        
        return themes
    
    def _detect_year_themes(self, movies_df, user_to_movies) -> List[Dict]:
        """Detect year/era themes from Section A movies"""
        themes = []
        
        if 'title' not in movies_df.columns:
            return themes
        
        # Extract years from titles
        import re
        era_to_movies = {}
        
        for _, row in movies_df.iterrows():
            title = str(row.get('title', ''))
            year_match = re.search(r'\((\d{4})\)', title)
            if year_match:
                year = int(year_match.group(1))
                # Group into decades
                decade = (year // 10) * 10
                era = f"{decade}s"
                
                if era not in era_to_movies:
                    era_to_movies[era] = []
                era_to_movies[era].append(row['movieId'])
        
        # Build themes
        for era, movie_ids in era_to_movies.items():
            if len(movie_ids) >= 2:
                user_movies = {}
                for uid, umovies in user_to_movies.items():
                    common = set(umovies) & set(movie_ids)
                    if common:
                        user_movies[uid] = list(common)
                
                if len(user_movies) >= 2:
                    themes.append({
                        'theme_type': 'year',
                        'theme_value': era,
                        'user_movies': user_movies,
                        'score': len(user_movies)
                    })
        
        return themes
    
    def _select_top_themes(self, themes: List[Dict], min_themes: int = 5, 
                          max_themes: int = 8) -> List[Dict]:
        """
        Select top themes ensuring diversity across theme types.
        
        Strategy:
        - Prioritize themes with highest user support
        - Ensure at least one theme from each type if possible
        - Limit to max_themes total
        """
        if not themes:
            return []
        
        # Sort by score (descending)
        themes.sort(key=lambda x: x['score'], reverse=True)
        
        # Ensure diversity: pick at least one from each type
        selected = []
        used_types = set()
        
        # First pass: one from each type
        for theme in themes:
            if theme['theme_type'] not in used_types:
                selected.append(theme)
                used_types.add(theme['theme_type'])
                if len(selected) >= max_themes:
                    break
        
        # Second pass: fill remaining slots with highest scores
        if len(selected) < min_themes:
            for theme in themes:
                if theme not in selected:
                    selected.append(theme)
                    if len(selected) >= max_themes:
                        break
        
        print(f"[THEMES] Selected {len(selected)} themes: {[t['theme_type'] for t in selected]}")
        return selected[:max_themes]
    
    def _generate_theme_block_enhanced(self, theme: Dict, group_users: List[int],
                                      excluded_ids: Set[int]) -> Dict[str, Any]:
        """
        Generate enhanced theme block with proper user attribution.
        
        Args:
            theme: Dict with theme_type, theme_value, user_movies, score
            group_users: List of user IDs
            excluded_ids: Set of movie IDs to exclude
        """
        theme_type = theme['theme_type']
        theme_value = theme['theme_value']
        user_movies = theme['user_movies']
        
        # Format theme name
        if theme_type == 'director':
            theme_name = f"Films by {theme_value}"
        elif theme_type == 'actor':
            theme_name = f"Featuring {theme_value}"
        elif theme_type == 'keyword':
            theme_name = f"Exploring {theme_value.title()}"
        elif theme_type == 'country':
            theme_name = f"Cinema from {theme_value}"
        elif theme_type == 'genre':
            theme_name = f"{theme_value} Films"
        elif theme_type == 'year':
            theme_name = f"Classics from the {theme_value}"
        else:
            theme_name = theme_value
        
        # Find matching movies (not in excluded set)
        matching_movies = self._find_movies_for_theme(theme, excluded_ids)
        
        if not matching_movies:
            return None
        
        # Generate justification citing specific users
        user_ids = list(user_movies.keys())
        if len(user_ids) == 1:
            justification = f"User {user_ids[0]} enjoyed this theme"
        elif len(user_ids) == 2:
            justification = f"Users {user_ids[0]} and {user_ids[1]} both appreciated this theme"
        else:
            justification = f"Users {', '.join(map(str, user_ids[:2]))} and {len(user_ids)-2} others enjoyed this theme"
        
        return {
            'theme_name': theme_name,
            'theme_type': theme_type,
            'justification': justification,
            'user_count': len(user_ids),
            'recommended_movies': matching_movies[:3]  # Top 3 per theme
        }
    
    def _find_movies_for_theme(self, theme: Dict, excluded_ids: Set[int]) -> List[Dict]:
        """Find movies matching the theme criteria"""
        theme_type = theme['theme_type']
        theme_value = theme['theme_value']
        
        # Filter movies by theme
        if theme_type == 'director':
            matching = self.movies_df[
                self.movies_df['Director'].str.contains(theme_value, case=False, na=False)
            ]
        elif theme_type == 'actor':
            matching = self.movies_df[
                self.movies_df['Actors'].str.contains(theme_value, case=False, na=False)
            ]
        elif theme_type == 'keyword':
            matching = self.movies_df[
                self.movies_df['Keywords'].str.contains(theme_value, case=False, na=False)
            ]
        elif theme_type == 'country':
            matching = self.movies_df[
                self.movies_df['Production_Countries'].str.contains(theme_value, case=False, na=False)
            ]
        elif theme_type == 'genre':
            # Restore original genre format
            original_genre = theme_value.replace(' + ', '|')
            matching = self.movies_df[
                self.movies_df['genres'] == original_genre
            ]
        elif theme_type == 'year':
            # Extract decade
            decade = int(theme_value.replace('s', ''))
            matching = self.movies_df[
                self.movies_df['title'].str.contains(f'\\(({decade}\\d)\\)', regex=True, na=False)
            ]
        else:
            return []
        
        # Exclude already recommended
        matching = matching[~matching['movieId'].isin(excluded_ids)]
        
        # Return top 3
        results = []
        for _, row in matching.head(3).iterrows():
            results.append({
                'movie_id': row['movieId'],
                'title': row['title']
            })
        
        return results
    
    def _extract_features_from_explanations(self, 
                                           section_a: List[Dict[str, Any]]) -> List[str]:
        """
        Extract features from explanation text using regex patterns.
        
        Patterns:
        - "focus on (.*)"
        - "themes like (.*)"
        """
        all_features = []
        
        for rec in section_a:
            user_explanations = rec.get('user_explanations', {})
            
            for uid, expl in user_explanations.items():
                # Check primary and secondary reasons
                reasons = [expl.get('primary_reason', '')]
                reasons.extend(expl.get('secondary_reasons', []))
                
                for reason in reasons:
                    # Pattern 1: "focus on (.*)"
                    match1 = re.search(r'focus on (.*?)(?:\.|,|$)', reason)
                    if match1:
                        features_str = match1.group(1)
                        features = [f.strip().title() for f in features_str.split(',')]
                        all_features.extend(features)
                    
                    # Pattern 2: "themes like (.*)"
                    match2 = re.search(r'themes like (.*?)(?:\.|,|$)', reason)
                    if match2:
                        features_str = match2.group(1)
                        features = [f.strip().title() for f in features_str.split(',')]
                        all_features.extend(features)
        
        return all_features
    
    def _generate_theme_block(self, feature: str, count: int, 
                              total_users: int, group_users: List[int],
                              excluded_ids: Set[int]) -> Dict[str, Any]:
        """
        Generate a thematic recommendation block for a specific feature.
        
        Args:
            feature: The feature name (e.g., "Christopher Nolan")
            count: Number of users who share this feature
            total_users: Total number of users in group
            group_users: List of user IDs
            excluded_ids: Set of movie IDs to exclude
        
        Returns:
            Theme block dict or None if no valid movies found
        """
        # Determine theme type
        theme_type = self._classify_feature_type(feature)
        
        # Generate theme title
        theme_title = self._generate_theme_title(feature, theme_type)
        
        # Generate explanation basis
        explanation_basis = f"Detected in explanations of {count} out of {total_users} members."
        
        # Find movies matching this feature
        matching_movies = self._find_movies_by_feature(
            feature, theme_type, excluded_ids
        )
        
        if not matching_movies:
            return None
        
        # Score and rank matching movies
        scored_movies = []
        for movie_id in matching_movies[:20]:  # Limit search space
            # Use Hybrid Model 1 to score
            scores = []
            for uid in group_users:
                score = self.h1.predict(uid, movie_id)
                if not np.isnan(score):
                    scores.append(score)
            
            if scores:
                avg_score = np.mean(scores)
                scored_movies.append((movie_id, avg_score))
        
        # Sort by score
        scored_movies.sort(key=lambda x: x[1], reverse=True)
        
        # Select top 3-5 movies
        top_movies = scored_movies[:5]
        
        if not top_movies:
            return None
        
        # Generate movie entries
        recommended_movies = []
        for movie_id, score in top_movies:
            title = self._get_movie_title(movie_id)
            explanation = self._generate_theme_movie_explanation(
                feature, theme_type
            )
            
            recommended_movies.append({
                'movie_id': movie_id,
                'title': title,
                'explanation': explanation
            })
        
        return {
            'theme_title': theme_title,
            'theme_type': theme_type,
            'theme_value': feature,
            'explanation_basis': explanation_basis,
            'recommended_movies': recommended_movies
        }
    
    def _classify_feature_type(self, feature: str) -> str:
        """
        Classify feature type based on content.
        
        Heuristics:
        - If contains common director names → DIRECTOR
        - If contains common actor names → ACTOR
        - If all lowercase or contains spaces → KEYWORD/THEME
        """
        # Simple heuristic: if capitalized and looks like a name → DIRECTOR/ACTOR
        # Otherwise → KEYWORD/THEME
        
        # Check if it's a person name (simple heuristic)
        words = feature.split()
        if len(words) >= 2 and all(w[0].isupper() for w in words if w):
            # Likely a person name
            # Could be director or actor - default to DIRECTOR for now
            return 'DIRECTOR'
        
        # Otherwise, it's a keyword/theme
        return 'KEYWORD'
    
    def _generate_theme_title(self, feature: str, theme_type: str) -> str:
        """Generate human-readable theme title."""
        if theme_type == 'DIRECTOR':
            return f"The group shows a shared interest in {feature} films."
        elif theme_type == 'ACTOR':
            return f"The group shows a shared interest in {feature} performances."
        else:  # KEYWORD/THEME
            return f"The group shows a shared interest in {feature}."
    
    def _generate_theme_movie_explanation(self, feature: str, 
                                         theme_type: str) -> str:
        """Generate explanation for a movie in theme block."""
        if theme_type == 'DIRECTOR':
            return f"Recommended for its focus on {feature}."
        elif theme_type == 'ACTOR':
            return f"Recommended for its focus on {feature}."
        else:  # KEYWORD/THEME
            return f"Recommended for its focus on {feature}."
    
    def _find_movies_by_feature(self, feature: str, theme_type: str,
                                excluded_ids: Set[int]) -> List[int]:
        """
        Find movies matching a specific feature.
        
        Search in:
        - Directors (if theme_type == DIRECTOR)
        - Actors (if theme_type == ACTOR)
        - Keywords (if theme_type == KEYWORD)
        """
        matching_movies = []
        
        for idx, row in self.movies_df.iterrows():
            movie_id = row['movieId']
            
            # Skip excluded movies
            if movie_id in excluded_ids:
                continue
            
            # Search based on theme type
            if theme_type == 'DIRECTOR':
                # Check directors column if exists
                if 'directors' in row and pd.notna(row['directors']):
                    directors_str = str(row['directors']).lower()
                    if feature.lower() in directors_str:
                        matching_movies.append(movie_id)
            
            elif theme_type == 'ACTOR':
                # Check actors column if exists
                if 'actors' in row and pd.notna(row['actors']):
                    actors_str = str(row['actors']).lower()
                    if feature.lower() in actors_str:
                        matching_movies.append(movie_id)
            
            else:  # KEYWORD/THEME
                # Check keywords column if exists
                if 'keywords' in row and pd.notna(row['keywords']):
                    keywords_str = str(row['keywords']).lower()
                    if feature.lower() in keywords_str:
                        matching_movies.append(movie_id)
        
        return matching_movies
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def _get_watched_set(self, group_users: List[int]) -> Set[int]:
        """Get set of all movies watched by any group member."""
        watched = set()
        for uid in group_users:
            if uid in self.cf_matrix.index:
                user_watched = self.cf_matrix.loc[uid].dropna().index.tolist()
                watched.update(user_watched)
        return watched
    
    def _get_direct_watchlist_set(self, group_users: List[int]) -> Set[int]:
        """Get set of all movies in any group member's watchlist."""
        wl_subset = self.watchlist_df[
            self.watchlist_df['userId'].isin(group_users)
        ]
        return set(wl_subset['movieId'].unique())
    
    def _get_movie_title(self, movie_id: int) -> str:
        """Get movie title by ID."""
        rows = self.movies_df[self.movies_df['movieId'] == movie_id]
        if not rows.empty:
            return rows.iloc[0]['title']
        return f"Unknown Movie ({movie_id})"
