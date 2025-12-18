"""
Temporal Preference Analyzer
-----------------------------
Analyzes users' temporal viewing patterns to ensure recommendations
match their historical preferences regarding movie release years.

Key Features:
- Detects if a user prefers recent vs classic movies
- Calculates temporal preference scores
- Filters recommendations based on temporal compatibility
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime


class TemporalPreferenceAnalyzer:
    """
    Analyzes temporal patterns in user viewing history to improve recommendations.
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
        self.enriched_ratings = self.ratings.merge(
            self.movies[['movieId', 'year']], 
            on='movieId', 
            how='left'
        )
        
        # Calculate current year for recency calculations
        self.current_year = datetime.now().year
        
        # Cache for user profiles
        self._user_profiles = {}
    
    def get_user_temporal_profile(self, user_id: int) -> Dict:
        """
        Analyze a user's temporal viewing preferences.
        
        Returns:
            Dict with keys:
                - 'avg_movie_age': Average age of movies watched (in years)
                - 'preference_type': 'recent', 'classic', 'mixed'
                - 'year_range': (min_year, max_year) watched
                - 'recency_score': 0-1, higher means prefers newer movies
                - 'min_acceptable_year': Minimum year to recommend
                - 'max_acceptable_year': Maximum year to recommend
        """
        if user_id in self._user_profiles:
            return self._user_profiles[user_id]
        
        user_ratings = self.enriched_ratings[
            self.enriched_ratings['userId'] == user_id
        ].copy()
        
        if len(user_ratings) == 0:
            # No history, return neutral profile
            return self._neutral_profile()
        
        # Filter out movies with missing years
        user_ratings = user_ratings[user_ratings['year'] > 0]
        
        if len(user_ratings) == 0:
            return self._neutral_profile()
        
        # Calculate movie ages at time of rating
        user_ratings['rating_timestamp'] = pd.to_datetime(
            user_ratings['timestamp'], 
            unit='s', 
            errors='coerce'
        )
        user_ratings['rating_year'] = user_ratings['rating_timestamp'].dt.year
        user_ratings['movie_age_at_rating'] = (
            user_ratings['rating_year'] - user_ratings['year']
        )
        
        # Calculate statistics
        avg_movie_age = user_ratings['movie_age_at_rating'].mean()
        std_movie_age = user_ratings['movie_age_at_rating'].std()
        min_year = user_ratings['year'].min()
        max_year = user_ratings['year'].max()
        
        # Calculate recency score (0 = only old movies, 1 = only new movies)
        # Based on average age: 0 years old = 1.0, 50+ years old = 0.0
        recency_score = max(0, min(1, 1 - (avg_movie_age / 50)))
        
        # Determine preference type
        if avg_movie_age < 5:
            preference_type = 'recent'
        elif avg_movie_age > 20:
            preference_type = 'classic'
        else:
            preference_type = 'mixed'
        
        # Calculate acceptable year range for recommendations
        # Use mean ± 1.5 * std as acceptable range
        if std_movie_age > 0:
            year_tolerance = 1.5 * std_movie_age
        else:
            year_tolerance = 10  # Default tolerance
        
        # Calculate based on movie ages, then convert to years
        min_acceptable_age = max(0, avg_movie_age - year_tolerance)
        max_acceptable_age = avg_movie_age + year_tolerance
        
        # Convert ages back to years
        min_acceptable_year = int(self.current_year - max_acceptable_age)
        max_acceptable_year = int(self.current_year - min_acceptable_age)
        
        # Ensure reasonable bounds
        min_acceptable_year = max(1900, min_acceptable_year)
        max_acceptable_year = min(self.current_year + 2, max_acceptable_year)
        
        profile = {
            'avg_movie_age': float(avg_movie_age),
            'std_movie_age': float(std_movie_age),
            'preference_type': preference_type,
            'year_range': (int(min_year), int(max_year)),
            'recency_score': float(recency_score),
            'min_acceptable_year': min_acceptable_year,
            'max_acceptable_year': max_acceptable_year,
            'total_ratings': len(user_ratings)
        }
        
        self._user_profiles[user_id] = profile
        return profile
    
    def _neutral_profile(self) -> Dict:
        """Return a neutral profile for users with no history."""
        return {
            'avg_movie_age': 15.0,
            'std_movie_age': 15.0,
            'preference_type': 'mixed',
            'year_range': (1980, self.current_year),
            'recency_score': 0.5,
            'min_acceptable_year': 1980,
            'max_acceptable_year': self.current_year + 2,
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
        
        Args:
            user_id: User ID
            movie_id: Movie ID
            strict: If True, use stricter filtering
            
        Returns:
            True if movie is temporally compatible
        """
        profile = self.get_user_temporal_profile(user_id)
        
        # Get movie year
        movie_year = self.movies[
            self.movies['movieId'] == movie_id
        ]['year'].values
        
        if len(movie_year) == 0 or movie_year[0] == 0:
            # Unknown year, allow it (neutral)
            return True
        
        movie_year = int(movie_year[0])
        
        if strict:
            # Strict mode: must be within acceptable range
            return (
                profile['min_acceptable_year'] <= movie_year <= 
                profile['max_acceptable_year']
            )
        else:
            # Lenient mode: allow some flexibility
            # Expand range by 50%
            tolerance = (
                profile['max_acceptable_year'] - 
                profile['min_acceptable_year']
            ) * 0.5
            
            min_year = profile['min_acceptable_year'] - tolerance
            max_year = profile['max_acceptable_year'] + tolerance
            
            return min_year <= movie_year <= max_year
    
    def get_temporal_compatibility_score(
        self, 
        user_id: int, 
        movie_id: int
    ) -> float:
        """
        Calculate a compatibility score (0-1) based on temporal preferences.
        
        Returns:
            Float between 0 (incompatible) and 1 (perfect match)
        """
        profile = self.get_user_temporal_profile(user_id)
        
        # Get movie year
        movie_year = self.movies[
            self.movies['movieId'] == movie_id
        ]['year'].values
        
        if len(movie_year) == 0 or movie_year[0] == 0:
            # Unknown year, return neutral score
            return 0.5
        
        movie_year = int(movie_year[0])
        movie_age = self.current_year - movie_year
        
        # Calculate distance from user's average preference
        avg_age = profile['avg_movie_age']
        std_age = profile['std_movie_age']
        
        if std_age == 0:
            std_age = 10  # Default
        
        # Use Gaussian distribution
        # Score is highest when movie age matches user's average
        distance = abs(movie_age - avg_age)
        score = np.exp(-(distance ** 2) / (2 * (std_age ** 2)))
        
        return float(score)
    
    def get_temporal_compatibility_batch(
        self,
        user_id: int,
        movie_ids: List[int]
    ) -> Dict[int, float]:
        """
        OPTIMIZED: Vectorized temporal compatibility for multiple movies.
        
        Complexity: O(n+m) instead of O(n*m) where n=users, m=movies
        
        Args:
            user_id: User ID
            movie_ids: List of movie IDs
            
        Returns:
            Dict mapping movie_id -> compatibility_score
        """
        profile = self.get_user_temporal_profile(user_id)
        
        # Get all movie years at once (vectorized)
        movie_years_df = self.movies[
            self.movies['movieId'].isin(movie_ids)
        ][['movieId', 'year']].copy()
        
        # Filter out movies with missing years
        movie_years_df = movie_years_df[movie_years_df['year'] > 0]
        
        if movie_years_df.empty:
            # All unknown years, return neutral scores
            return {mid: 0.5 for mid in movie_ids}
        
        # Calculate ages (vectorized)
        movie_years_df['age'] = self.current_year - movie_years_df['year']
        
        # Calculate distances from user's average preference
        avg_age = profile['avg_movie_age']
        std_age = profile['std_movie_age'] if profile['std_movie_age'] > 0 else 10
        
        # Vectorized Gaussian calculation
        movie_years_df['distance'] = np.abs(movie_years_df['age'] - avg_age)
        movie_years_df['score'] = np.exp(
            -(movie_years_df['distance'] ** 2) / (2 * (std_age ** 2))
        )
        
        # Create result dictionary
        result = dict(zip(movie_years_df['movieId'], movie_years_df['score']))
        
        # Fill in missing movies with neutral score
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
        Filter and score candidate movies by temporal compatibility.
        
        OPTIMIZED: Uses batch processing for better performance.
        
        Args:
            user_id: User ID
            candidate_movies: List of movie IDs to filter
            min_score: Minimum compatibility score to keep
            strict: Use strict filtering
            
        Returns:
            List of (movie_id, temporal_score) tuples, sorted by score
        """
        # Use batch processing
        scores = self.get_temporal_compatibility_batch(user_id, candidate_movies)
        
        results = []
        
        for movie_id, score in scores.items():
            if strict:
                # Check strict compatibility
                if not self.is_movie_temporally_compatible(
                    user_id, movie_id, strict=True
                ):
                    continue
            
            if score >= min_score:
                results.append((movie_id, score))
        
        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results
    
    def get_group_temporal_profile(self, group_user_ids: List[int]) -> Dict:
        """
        Aggregate temporal profiles for a group of users.
        
        Returns:
            Dict with aggregated group preferences
        """
        profiles = [
            self.get_user_temporal_profile(uid) 
            for uid in group_user_ids
        ]
        
        if not profiles:
            return self._neutral_profile()
        
        # Aggregate statistics
        avg_ages = [p['avg_movie_age'] for p in profiles]
        recency_scores = [p['recency_score'] for p in profiles]
        
        # Use most restrictive range (intersection)
        min_years = [p['min_acceptable_year'] for p in profiles]
        max_years = [p['max_acceptable_year'] for p in profiles]
        
        group_min_year = max(min_years)  # Most restrictive minimum
        group_max_year = min(max_years)  # Most restrictive maximum
        
        # If ranges don't overlap, use average
        if group_min_year > group_max_year:
            group_min_year = int(np.mean(min_years))
            group_max_year = int(np.mean(max_years))
        
        group_avg_age = np.mean(avg_ages)
        group_recency = np.mean(recency_scores)
        
        # Determine group preference type
        if group_recency > 0.7:
            pref_type = 'recent'
        elif group_recency < 0.3:
            pref_type = 'classic'
        else:
            pref_type = 'mixed'
        
        return {
            'avg_movie_age': float(group_avg_age),
            'preference_type': pref_type,
            'recency_score': float(group_recency),
            'min_acceptable_year': group_min_year,
            'max_acceptable_year': group_max_year,
            'group_size': len(group_user_ids),
            'individual_profiles': profiles
        }
    
    def explain_temporal_mismatch(
        self, 
        user_id: int, 
        movie_id: int
    ) -> str:
        """
        Generate a human-readable explanation for why a movie doesn't fit
        the user's temporal preferences.
        """
        profile = self.get_user_temporal_profile(user_id)
        
        movie_year = self.movies[
            self.movies['movieId'] == movie_id
        ]['year'].values
        
        if len(movie_year) == 0 or movie_year[0] == 0:
            return "Movie year unknown"
        
        movie_year = int(movie_year[0])
        movie_age = self.current_year - movie_year
        
        if profile['preference_type'] == 'recent':
            if movie_age > 10:
                return (
                    f"This movie from {movie_year} may be too old. "
                    f"You typically watch movies from the last "
                    f"{int(profile['avg_movie_age'])} years."
                )
        elif profile['preference_type'] == 'classic':
            if movie_age < 5:
                return (
                    f"This movie from {movie_year} may be too recent. "
                    f"You typically prefer older films "
                    f"(average age: {int(profile['avg_movie_age'])} years)."
                )
        
        return (
            f"This movie from {movie_year} is outside your typical range "
            f"({profile['min_acceptable_year']}-{profile['max_acceptable_year']})."
        )
