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
                 enable_temporal_filtering=True):
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
        """
        self.h1 = hybrid_model_1
        self.h2 = hybrid_model_2
        self.h3 = hybrid_model_3
        self.movies_df = movies_df
        self.watchlist_df = watchlist_df
        self.cf_matrix = cf_matrix
        self.enable_temporal_filtering = enable_temporal_filtering
        
        # Initialize temporal analyzer if ratings data is provided
        if ratings_df is not None and enable_temporal_filtering:
            self.temporal_analyzer = TemporalPreferenceAnalyzer(ratings_df, movies_df)
        else:
            self.temporal_analyzer = None
        
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
        
        # Generate Section A
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
        Generate shared interest themes section.
        
        STRICT RULES:
        - Explanation-first approach
        - Derive interests from EXISTING explanations
        - DO NOT infer from raw metadata
        - DO NOT use high-level genres
        - Valid types: Directors, Keywords, Themes, Actors
        """
        # 1. Extract features from Section A explanations
        all_features = self._extract_features_from_explanations(section_a)
        
        # 2. Count and normalize features
        feature_counts = Counter(all_features)
        
        # 3. Select top 3-4 features that meet consensus criteria
        consensus_threshold = max(2, len(group_users) // 2)
        
        selected_features = [
            (feature, count) for feature, count in feature_counts.most_common()
            if count >= consensus_threshold
        ][:4]  # Max 4 themes
        
        if not selected_features:
            # Fallback: no strong consensus
            return []
        
        # 4. For each selected feature, generate thematic block
        section_c_output = []
        
        # Collect already recommended movie IDs
        excluded_ids = watched_set.copy()
        excluded_ids.update({rec['movie_id'] for rec in section_a})
        excluded_ids.update({rec['movie_id'] for rec in section_b})
        
        for feature, count in selected_features:
            theme_block = self._generate_theme_block(
                feature, count, len(group_users), 
                group_users, excluded_ids
            )
            
            if theme_block:
                section_c_output.append(theme_block)
                # Update excluded IDs to prevent duplicates across themes
                excluded_ids.update({
                    m['movie_id'] for m in theme_block['recommended_movies']
                })
        
        return section_c_output
    
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
