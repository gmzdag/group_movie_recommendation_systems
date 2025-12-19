"""
Temporal Preference Analyzer
-----------------------------
Analyzes users' temporal viewing patterns to ensure recommendations
match their historical preferences regarding movie release years.

Key Features:
- Detects if a user prefers movies from specific eras (e.g. 90s, 2020s)
- Calculates temporal preference scores based on release year distribution
- Filters recommendations based on temporal compatibility
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime


class TemporalPreferenceAnalyzer:
    """
    Analyzes temporal patterns in user viewing history to improve recommendations.
    Uses the distribution of release years of movies the user liked.
    """
    
    def __init__(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame):
        """
        Initialize the temporal analyzer.
        
        Args:
            ratings_df: DataFrame with columns [userId, movieId, rating, timestamp]
            movies_df: DataFrame with columns [movieId, title, year, ...]
        """
        self.ratings = ratings_df.copy()
        self.movies = movies_df.copy()
        
        # Merge to get movie years with ratings
        # Use inner join or left join, but we need years
        self.enriched_ratings = self.ratings.merge(
            self.movies[['movieId', 'year']], 
            on='movieId', 
            how='left'
        )
        
        # Calculate current year
        self.current_year = datetime.now().year
        
        # Cache for user profiles
        self._user_profiles = {}
    
    def get_user_temporal_profile(self, user_id: int) -> Dict:
        """
        Analyze a user's temporal viewing preferences based on Release Years.
        
        Returns:
            Dict with keys:
                - 'mean_release_year': Weighted average release year of liked movies
                - 'std_release_year': Standard deviation of release years
                - 'preference_type': 'modern', 'classic', 'era_specific', 'broad'
                - 'year_range': (min_year, max_year) of interest
                - 'recency_bias': deviation from current year
        """
        if user_id in self._user_profiles:
            return self._user_profiles[user_id]
        
        # Get user ratings
        user_ratings = self.enriched_ratings[
            self.enriched_ratings['userId'] == user_id
        ].copy()
        
        if len(user_ratings) == 0:
            return self._neutral_profile()
        
        # Filter out movies with missing years and keep only "liked" movies (rating >= 3.0)
        # We consider 3.0 as neutral/positive enough to signal interest in that era
        valid_ratings = user_ratings[
            (user_ratings['year'] > 1900) & 
            (user_ratings['rating'] >= 3.0)
        ]
        
        if len(valid_ratings) < 5:
            # Fallback to all ratings if not enough positive ones
            valid_ratings = user_ratings[user_ratings['year'] > 1900]
            
        if len(valid_ratings) == 0:
            return self._neutral_profile()
            
        # Weighted statistics based on rating
        # Higher rated movies contribute more to the mean year
        years = valid_ratings['year'].values
        weights = valid_ratings['rating'].values
        
        # Weighted Mean
        mean_year = np.average(years, weights=weights)
        
        # Weighted Variance/Std
        variance = np.average((years - mean_year)**2, weights=weights)
        std_year = np.sqrt(variance)
        
        # Determine preference type
        current_era_threshold = self.current_year - 5
        
        if mean_year >= current_era_threshold:
            preference_type = 'modern'  # Likes very recent stuff
        elif std_year < 5:
            preference_type = 'era_specific' # Focused on a specific time (e.g. only 90s)
        elif std_year > 15:
            preference_type = 'broad' # Watches everything
        else:
            preference_type = 'classic' # Generally older movies
            
        # Calculate acceptance range (Mean +/- 2 STD, clamped)
        # We assume a minimum std of 5 years to avoid being too restrictive
        effective_std = max(5.0, std_year)
        
        min_acceptable = int(mean_year - (2.0 * effective_std))
        max_acceptable = int(mean_year + (2.0 * effective_std))
        
        # Clamp to realistic values
        min_acceptable = max(1900, min_acceptable)
        max_acceptable = min(self.current_year + 1, max_acceptable)
        
        profile = {
            'mean_release_year': float(mean_year),
            'std_release_year': float(std_year),
            'preference_type': preference_type,
            'year_range': (min_acceptable, max_acceptable),
            'total_ratings': len(valid_ratings)
        }
        
        self._user_profiles[user_id] = profile
        return profile
    
    def _neutral_profile(self) -> Dict:
        """Return a neutral profile for users with no history."""
        return {
            'mean_release_year': float(self.current_year - 15), # ~2010
            'std_release_year': 20.0,
            'preference_type': 'broad',
            'year_range': (1970, self.current_year + 1),
            'total_ratings': 0
        }
    
    def is_movie_temporally_compatible(
        self, 
        user_id: int, 
        movie_id: int,
        strict: bool = False
    ) -> bool:
        """
        Check if a movie's release year is compatible with user's preferences.
        """
        profile = self.get_user_temporal_profile(user_id)
        
        movie_year = self.movies[
            self.movies['movieId'] == movie_id
        ]['year'].values
        
        if len(movie_year) == 0 or movie_year[0] == 0:
            return True # Unknown year, benefit of doubt
        
        year = int(movie_year[0])
        
        if strict:
            return profile['year_range'][0] <= year <= profile['year_range'][1]
        else:
            # Lenient: Expand range by 50% of width
            width = profile['year_range'][1] - profile['year_range'][0]
            margin = width * 0.25
            return (profile['year_range'][0] - margin) <= year <= (profile['year_range'][1] + margin)
    
    def get_temporal_compatibility_score(
        self, 
        user_id: int, 
        movie_id: int
    ) -> float:
        """
        Calculate a compatibility score (0-1) based on release year Gaussian.
        """
        profile = self.get_user_temporal_profile(user_id)
        
        movie_year = self.movies[
            self.movies['movieId'] == movie_id
        ]['year'].values
        
        if len(movie_year) == 0 or movie_year[0] == 0:
            return 0.5
        
        year = int(movie_year[0])
        
        # Gaussian decay from mean_release_year
        mean = profile['mean_release_year']
        std = max(10.0, profile['std_release_year']) # Minimum spread of 10 years
        
        diff = abs(year - mean)
        score = np.exp(-(diff ** 2) / (2 * (std ** 2)))
        
        return float(score)
    
    def get_temporal_compatibility_batch(
        self,
        user_id: int,
        movie_ids: List[int]
    ) -> Dict[int, float]:
        """
        Vectorized temporal compatibility for multiple movies.
        """
        profile = self.get_user_temporal_profile(user_id)
        
        # Get movie years
        movie_years_df = self.movies[
            self.movies['movieId'].isin(movie_ids)
        ][['movieId', 'year']].copy()
        
        movie_years_df = movie_years_df[movie_years_df['year'] > 0]
        
        if movie_years_df.empty:
            return {mid: 0.5 for mid in movie_ids}
        
        mean = profile['mean_release_year']
        std = max(10.0, profile['std_release_year'])
        
        # Vectorized Gaussian
        movie_years_df['diff'] = np.abs(movie_years_df['year'] - mean)
        movie_years_df['score'] = np.exp(
            -(movie_years_df['diff'] ** 2) / (2 * (std ** 2))
        )
        
        result = dict(zip(movie_years_df['movieId'], movie_years_df['score']))
        
        # Fill missing
        for mid in movie_ids:
            if mid not in result:
                result[mid] = 0.5
                
        return result

    def filter_recommendations_by_temporal_fit(
        self,
        user_id: int,
        candidate_movies: List[int],
        min_score: float = 0.3,
        strict: bool = False
    ) -> List[Tuple[int, float]]:
        """
        Filter recommendations based on temporal fit.
        """
        scores = self.get_temporal_compatibility_batch(user_id, candidate_movies)
        filtered = []
        
        for mid, score in scores.items():
            if strict and not self.is_movie_temporally_compatible(user_id, mid, strict=True):
                continue
            if score >= min_score:
                filtered.append((mid, score))
                
        return sorted(filtered, key=lambda x: x[1], reverse=True)

    def get_group_temporal_profile(self, group_user_ids: List[int]) -> Dict:
        """
        Aggregate temporal profiles for a group.
        """
        profiles = [self.get_user_temporal_profile(uid) for uid in group_user_ids]
        
        if not profiles:
            return self._neutral_profile()
            
        # Average the means
        means = [p['mean_release_year'] for p in profiles]
        group_mean = np.mean(means)
        
        # Union of ranges
        min_years = [p['year_range'][0] for p in profiles]
        max_years = [p['year_range'][1] for p in profiles]
        
        # Use a "middle ground" range
        group_min = np.mean(min_years)
        group_max = np.mean(max_years)
        
        return {
            'mean_release_year': group_mean,
            'year_range': (int(group_min), int(group_max)),
            'individual_profiles': profiles
        }
        
    def explain_temporal_mismatch(
        self, 
        user_id: int, 
        movie_id: int
    ) -> str:
        """
        Generate explanation for temporal mismatch.
        """
        profile = self.get_user_temporal_profile(user_id)
        movie_year_vals = self.movies[self.movies['movieId'] == movie_id]['year'].values
        
        if len(movie_year_vals) == 0:
            return "Year unknown"
            
        year = int(movie_year_vals[0])
        mean = int(profile['mean_release_year'])
        
        diff = year - mean
        
        if diff < -15:
            return f"From {year}, which is older than your typical preference (around {mean})."
        elif diff > 15:
            return f"From {year}, which is newer than your typical preference (around {mean})."
        else:
            return f"From {year}, slightly outside your core preference range."
