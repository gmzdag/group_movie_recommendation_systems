"""
Content-Based Filtering Model (Signal Provider)
-----------------------------------------------
Generates similarity signals based on movie metadata (Genres, Overview, Cast, etc.)
Strictly designed for Hybrid System integration.

Features:
- Values: Genres, Overview, Keywords, Director, Actors
- Vectorizer: TF-IDF (1-2 ngrams)
- Profile: Weighted Centroid (Rating - Mean)
- Output: Cosine Similarity (-1.0 to 1.0)
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedModel:
    def __init__(self, movies_df, ratings_df):
        """
        Initialize and fit the Content-Based Model.
        
        Args:
            movies_df: DataFrame containing metadata (movieId, genres, etc.)
            ratings_df: DataFrame containing user history (userId, movieId, rating)
        """
        print(f"\n[DEBUG] Initializing ContentBasedModel (Signal Mode)...")
        
        self.movies_df = movies_df.copy()
        self.ratings_df = ratings_df.copy()
        
        # 1. Prepare Data (Text Soup)
        print(f"[DEBUG] Constructing 'text soup' from metadata...")
        self.movies_df['soup'] = self._create_soup(self.movies_df)
        
        # 2. Vectorize
        print(f"[DEBUG] Fitting TF-IDF Vectorizer...")
        self.tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), min_df=2)
        self.tfidf_matrix = self.tfidf.fit_transform(self.movies_df['soup'])
        
        print(f"[DEBUG] TF-IDF Matrix shape: {self.tfidf_matrix.shape}")
        
        # 3. Create Mappings for fast access
        # Mapping movieId -> Matrix Index
        self.movie_to_idx = pd.Series(
            self.movies_df.index, index=self.movies_df['movieId']
        ).to_dict()
        
        # Pre-compute User Means for Profile Construction efficiently
        print(f"[DEBUG] Pre-computing user means...")
        self.user_means = self.ratings_df.groupby('userId')['rating'].mean()
        
        # Cache for User Profiles (Lazy Loading)
        self.user_profiles = {}
        
    def _create_soup(self, df):
        """
        Creates a single string 'soup' for each movie used for vectorization.
        Refined Weighting Strategy:
        - Genres (x2): Foundation of similarity
        - Director (x2): Strongest single-token signal
        - Keywords (x2): Specific plot elements
        - Actors (x1): Top 3 actors
        - Overview (x1): General Context
        
        Result: Overlap in multiple categories (e.g. Director + Keywords) yields much higher scores
        """
        def clean_token(x):
            # "Christopher Nolan" -> "ChristopherNolan"
            if isinstance(x, str):
                return x.replace(" ", "").lower() # Lowercase for consistency
            return ""

        soup = []
        for _, row in df.iterrows():
            # 1. Genres (x2)
            genres = str(row.get('genres', '')).replace('|', ' ')
            genres = (genres + " ") * 2
            
            # 2. Director (x2) - Very specific
            director_val = str(row.get('Director', ''))
            director = (clean_token(director_val) + " ") * 2
            
            # 3. Actors (x1) - Top 3
            actors_raw = str(row.get('Actors', ''))
            if actors_raw and actors_raw.lower() != 'nan':
                 # Split, clean each, join
                 act_list = [clean_token(a) for a in actors_raw.split(',')[:3]]
                 actors = " ".join(act_list)
            else:
                actors = ""
                
            # 4. Keywords (x2) - Specific plot points
            # "time travel" -> "time travel" (keep spaces? No, usually keywords are phrases)
            # Strategy: Replace '|' with space. Repeat string twice.
            # Keywords often contain valuable phrases handled by ngram=(1,2)
            kw_raw = str(row.get('Keywords', '')).replace('|', ' ')
            keywords = (kw_raw + " ") * 2
            
            # 5. Overview (x1)
            overview = str(row.get('Overview', ''))
            
            # Combined String
            combined = f"{genres} {director} {actors} {keywords} {overview}"
            soup.append(combined)
            
        return soup

    def _get_user_profile(self, user_id):
        """
        Computes (or retrieves) the User's Weighted Centroid Profile.
        Vector P_u = Sum( (r_ui - mu_u) * V_i ) / Normalization
        """
        if user_id in self.user_profiles:
            return self.user_profiles[user_id]
        
        # Get user history
        history = self.ratings_df[self.ratings_df['userId'] == user_id]
        
        if history.empty:
            return None
            
        user_mean = self.user_means.get(user_id, 3.0)
        
        # Vectors and Weights
        user_vectors = []
        weights = []
        
        for _, row in history.iterrows():
            mid = row['movieId']
            rating = row['rating']
            
            if mid in self.movie_to_idx:
                idx = self.movie_to_idx[mid]
                vec = self.tfidf_matrix[idx]
                
                # Weight: Centered Rating
                # 5-star -> positive influence, 1-star -> negative influence
                weight = rating - user_mean
                
                user_vectors.append(vec)
                weights.append(weight)
        
        if not user_vectors:
            return None
            
        # Weighted Average
        # Stack vectors (sparse) -> convert to dense/array for math? 
        # Sparse arithmetic is better for memory.
        
        # Calculation: Sum(w * v)
        # scipy sparse matrices support scalar multiplication
        
        profile_vec = None
        
        for w, v in zip(weights, user_vectors):
            weighted_v = v.multiply(w)
            if profile_vec is None:
                profile_vec = weighted_v
            else:
                profile_vec = profile_vec + weighted_v
                
        # Optional: Normalize? 
        # TF-IDF vectors are usually L2 normalized. 
        # The resulting profile vector should ideally be normalized for Cosine Sim to be robust.
        # But sklearn cosine_similarity handles normalization internally if vectors are raw.
        # It's safer to keep it as direction vector.
        
        self.user_profiles[user_id] = profile_vec
        return profile_vec

    def predict(self, user_id, movie_id):
        """
        Predicts similarity score between user profile and movie content.
        
        Returns:
            float: Cosine Similarity (-1.0 to 1.0)
            NaN: If user profile cannot be built or movie unknown
        """
        # 1. Validate Movie
        if movie_id not in self.movie_to_idx:
            return np.nan
            
        # 2. Get User Profile
        user_profile = self._get_user_profile(user_id)
        if user_profile is None:
            return np.nan
            
        # 3. Get Movie Vector
        idx = self.movie_to_idx[movie_id]
        movie_vector = self.tfidf_matrix[idx]
        
        # 4. Compute Cosine Similarity
        # cosine_similarity accepts 2D arrays
        score = cosine_similarity(user_profile, movie_vector)[0][0]
        
        return float(score)
