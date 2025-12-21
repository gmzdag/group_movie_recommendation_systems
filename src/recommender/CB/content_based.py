"""
Content-Based Filtering Model (Signal Provider) - V2
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
    def __init__(self, movies_df, ratings_df, weights=None):
        """
        Initialize and fit the Content-Based Model.
        
        Args:
            movies_df: DataFrame containing metadata (movieId, genres, etc.)
            ratings_df: DataFrame containing user history (userId, movieId, rating)
            weights: Dictionary of feature weights (default if None)
        """
        print(f"\n[DEBUG] Initializing ContentBasedModel (Signal Mode)...")
        
        self.movies_df = movies_df.copy()
        self.ratings_df = ratings_df.copy()

        # Default Weights
        self.weights = weights if weights else {
            'genres': 2,
            'director': 2,
            'keywords': 2,
            'actors': 1,
            'year': 1,
            'overview': 1,
            'companies': 0, # Default to 0 (disabled) to match previous baseline unless specified
            'countries': 0
        }
        print(f"[DEBUG] Using weights: {self.weights}")
        
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
        Uses self.weights for token repetition.
        """
        def clean_token(x):
            if isinstance(x, str):
                return x.replace(" ", "").lower() 
            return ""

        soup = []
        for _, row in df.iterrows():
            # 1. Genres
            genres = str(row.get('genres', '')).replace('|', ' ')
            genres = (genres + " ") * self.weights.get('genres', 2)
            
            # 2. Director
            director_val = str(row.get('Director', ''))
            director = (clean_token(director_val) + " ") * self.weights.get('director', 2)
            
            # 3. Actors
            actors_raw = str(row.get('Actors', ''))
            if actors_raw and actors_raw.lower() != 'nan':
                 act_list = [clean_token(a) for a in actors_raw.split(',')[:3]]
                 actors = " ".join(act_list)
            else:
                actors = ""
            actors = (actors + " ") * self.weights.get('actors', 1)
                
            # 4. Keywords
            kw_raw = str(row.get('Keywords', '')).replace('|', ' ')
            keywords = (kw_raw + " ") * self.weights.get('keywords', 2)
            
            # 5. Production Companies
            companies_weight = self.weights.get('companies', 0)
            companies = ""
            if companies_weight > 0:
                comp_raw = str(row.get('Production_Companies', ''))
                # Clean: "Warner Bros. Pictures" -> "warnbros.pictures" helps uniqueness
                # But simple lower+nospace is fine: "warnerbros.pictures"
                if comp_raw and comp_raw.lower() != 'nan':
                    # Take top 2 companies
                    c_list = [clean_token(c) for c in comp_raw.split(',')[:2]]
                    companies = " ".join(c_list)
                    companies = (companies + " ") * companies_weight

            # 6. Production Countries
            countries_weight = self.weights.get('countries', 0)
            countries = ""
            if countries_weight > 0:
                count_raw = str(row.get('Production_Countries', ''))
                if count_raw and count_raw.lower() != 'nan':
                    # Take all
                    ct_list = [clean_token(c) for c in count_raw.split(',')]
                    countries = " ".join(ct_list)
                    countries = (countries + " ") * countries_weight

            # 7. Year
            year_reps = self.weights.get('year', 1)
            year_str = ""
            if year_reps > 0:
                year = row.get('year', 0)
                if year > 0:
                    year_token = str(int(year))
                    decade_token = "Decade" + str(int(year) // 10 * 10)
                    year_str = f"{year_token} {decade_token} " 
                    year_str = year_str * year_reps
            
            # 8. Overview
            overview_reps = self.weights.get('overview', 1)
            overview = str(row.get('Overview', ''))
            overview = (overview + " ") * overview_reps
            
            # Combined String
            combined = f"{genres} {director} {actors} {keywords} {companies} {countries} {year_str} {overview}"
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
        profile_vec = None
        
        for w, v in zip(weights, user_vectors):
            weighted_v = v.multiply(w)
            if profile_vec is None:
                profile_vec = weighted_v
            else:
                profile_vec = profile_vec + weighted_v
                
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
        score = cosine_similarity(user_profile, movie_vector)[0][0]
        
        return float(score)

    def predict_rating(self, user_id, movie_id, top_k=20):
        """
        Predicts rating (1-5) using Content-Based k-NN Regression.
        Formula: r_hat = mu_u + [ Sum(sim(i,j) * (r_uj - mu_u)) / Sum(|sim(i,j)|) ]
        
        Args:
            user_id: Target User
            movie_id: Target Movie
            top_k: Number of content-similar items to use
            
        Returns:
            float: Predicted Rating (1.0 - 5.0)
            NaN: If prediction impossible
        """
        # 1. Validation and Setup
        if movie_id not in self.movie_to_idx:
            return np.nan
            
        target_idx = self.movie_to_idx[movie_id]
        
        # 2. Get User History
        history = self.ratings_df[self.ratings_df['userId'] == user_id]
        if history.empty:
            return np.nan
            
        user_mean = self.user_means.get(user_id, 3.0)
        
        # 3. Compute Similarities with User History
        history_mids = history['movieId'].values
        history_ratings = history['rating'].values
        
        # Filter to known movies only
        valid_mask = [m in self.movie_to_idx for m in history_mids]
        history_mids = history_mids[valid_mask]
        history_ratings = history_ratings[valid_mask]
        
        if len(history_mids) == 0:
            return np.nan
            
        history_indices = [self.movie_to_idx[m] for m in history_mids]
        
        # Get Vectors
        target_vec = self.tfidf_matrix[target_idx]
        history_vecs = self.tfidf_matrix[history_indices]
        
        # Cosine Similarity
        similarities = cosine_similarity(target_vec, history_vecs).flatten()
        
        # 4. Select Top K Neighbors
        if len(similarities) > top_k:
            top_indices = np.argsort(similarities)[::-1][:top_k]
        else:
            top_indices = np.arange(len(similarities))
            
        # 5. Weighted Average
        sims_k = similarities[top_indices]
        ratings_k = history_ratings[top_indices]
        
        sum_sim = np.sum(np.abs(sims_k))
        
        if sum_sim == 0:
            return user_mean
        
        centered_ratings = ratings_k - user_mean
        weighted_sum = np.dot(sims_k, centered_ratings)
        pred_offset = weighted_sum / sum_sim
        
        prediction = user_mean + pred_offset
        
        return float(prediction)

    def get_explanation(self, user_id, movie_id, top_k=5):
        """
        Generates a content-based explanation for why a movie is recommended.
        Focuses on high-value matches (Director, Actors, Keywords) over generic Genres.
        """
        if user_id not in self.user_profiles:
            return {"score": 0, "features": []}

        # Get user's high-rated movies
        user_ratings = self.ratings_df[self.ratings_df['userId'] == user_id]
        liked_movies = user_ratings[user_ratings['rating'] >= 4.0]['movieId'].tolist()
        
        if not liked_movies:
             liked_movies = user_ratings['movieId'].tolist()
             
        # Find the most similar liked movie to the candidate
        target_idx = self.movie_to_idx.get(movie_id)
        if target_idx is None:
            return {"score": 0, "features": []}
            
        # Check similarity against recent liked movies
        recent_liked = liked_movies[-20:]
        best_match = None
        best_sim = -1
        
        target_vec = self.tfidf_matrix[target_idx]
        
        for pid in recent_liked:
            pidx = self.movie_to_idx.get(pid)
            if pidx is not None:
                p_vec = self.tfidf_matrix[pidx]
                sim = (target_vec @ p_vec.T)[0, 0]
                if sim > best_sim:
                    best_sim = sim
                    best_match = pid
        
        features = []
        if best_match:
            # Explain the pair "Candidate" vs "Best Match"
            pair_reason = self._explain_pair(movie_id, best_match)
            if pair_reason:
                features.append(f"Similar to favorite '{self.movies_df[self.movies_df['movieId'] == best_match]['title'].values[0]}': {pair_reason}")
        
        return {
            "score": best_sim, 
            "features": features
        }

    def get_shared_traits(self, movie_a_id, movie_b_id):
        """
        Returns a dictionary of shared traits between two movies.
        """
        def get_vals(mid, col):
            try:
                row = self.movies_df[self.movies_df['movieId'] == mid]
                if row.empty: return set()
                val = row[col].values[0]
                if pd.isna(val): return set()
                return set([x.strip() for x in str(val).replace('|', ',').split(',') if x.strip()])
            except:
                return set()

        traits = {
            'Directors': get_vals(movie_a_id, 'Director') & get_vals(movie_b_id, 'Director'),
            'Actors': get_vals(movie_a_id, 'Actors') & get_vals(movie_b_id, 'Actors'),
            'Keywords': get_vals(movie_a_id, 'Keywords') & get_vals(movie_b_id, 'Keywords'),
            'Genres': get_vals(movie_a_id, 'genres') & get_vals(movie_b_id, 'genres')
        }
        return traits

    def _explain_pair(self, movie_a, movie_b):
        """
        Helper for internal string formatting.
        Uses get_shared_traits.
        """
        traits = self.get_shared_traits(movie_a, movie_b)
        reasons = []
        
        if traits['Directors']: reasons.append(f"Director: {', '.join(traits['Directors'])}")
        if traits['Actors']: 
            reasons.append(f"Actors: {', '.join(list(traits['Actors'])[:2])}")
        if traits['Keywords']:
            reasons.append(f"Themes: {', '.join(list(traits['Keywords'])[:3])}")
            
        generic = {'Drama', 'Comedy', 'Thriller', 'Action', 'Adventure', 'Romance', 'Crime'}
        specific = traits['Genres'] - generic
        
        if specific:
            reasons.append(f"Genre: {', '.join(specific)}")
        elif traits['Genres'] and not reasons:
             reasons.append(f"Genre: {', '.join(traits['Genres'])}")
             
        if not reasons: reasons.append("Matches Metadata/Plot")
        
        return ", ".join(reasons)
