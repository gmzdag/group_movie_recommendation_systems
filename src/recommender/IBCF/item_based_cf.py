"""
Item-Based Collaborative Filtering
----------------------------------
predict(), recommend(), recommend_group()
"""

import numpy as np
import pandas as pd


class ItemBasedCF:
    def __init__(self, raw_um, norm_um, item_neighbors, movies, top_k=60):
        print(f"\n[DEBUG] Initializing ItemBasedCF...")
        print(f"[DEBUG] - raw_um shape: {raw_um.shape}")
        print(f"[DEBUG] - norm_um shape: {norm_um.shape}")
        print(f"[DEBUG] - item_neighbors count: {len(item_neighbors)}")
        print(f"[DEBUG] - movies count: {len(movies)}")
        print(f"[DEBUG] - top_k: {top_k}")
        
        self.raw_um = raw_um
        self.norm_um = norm_um
        self.item_neighbors = item_neighbors
        self.movies = movies
        self.top_k = top_k
        
        print(f"[DEBUG] Pre-computing user means for faster prediction...")
        self.user_means = self.raw_um.mean(axis=1)
        print(f"[DEBUG] User means - Min: {self.user_means.min():.2f}, Max: {self.user_means.max():.2f}")

    # ------------------------------------------------------------
    # Predict rating
    # ------------------------------------------------------------
    def predict(self, user_id, movie_id, verbose=False):
        """
        Predict rating for user_id on movie_id
        
        Args:
            user_id: User ID
            movie_id: Movie ID
            verbose: Print debug info for this prediction
        """
        if verbose:
            print(f"\n[PREDICT] User {user_id}, Movie {movie_id}")
        
        # Case 1: Movie has no neighbors
        if movie_id not in self.item_neighbors:
            fallback = self.user_means.get(user_id, 3.0)
            if verbose:
                print(f"  → No neighbors found, returning user mean: {fallback:.2f}")
            return fallback
        
        user_orig = self.raw_um.loc[user_id]
        user_norm = self.norm_um.loc[user_id]
        neighbors = self.item_neighbors[movie_id]
        
        if verbose:
            print(f"  → Movie has {len(neighbors)} neighbors")
        
        # Find rated neighbors
        rated_neighbors = [
            m for m in neighbors.keys()
            if pd.notna(user_orig.get(m))
        ]
        
        if verbose:
            print(f"  → User rated {len(rated_neighbors)} of these neighbors")
        
        # Case 2: No rated neighbors
        if len(rated_neighbors) == 0:
            fallback = self.user_means.get(user_id, 3.0)
            if verbose:
                print(f"  → No rated neighbors, returning user mean: {fallback:.2f}")
            return fallback
        
        # Weighted prediction
        weights = np.array([neighbors[m] for m in rated_neighbors])
        ratings = user_norm.loc[rated_neighbors].values
        
        if verbose:
            print(f"  → Weights range: [{weights.min():.3f}, {weights.max():.3f}]")
            print(f"  → Normalized ratings range: [{ratings.min():.3f}, {ratings.max():.3f}]")
        
        # Case 3: All ratings are zero (user rated everything at their mean)
        if np.all(ratings == 0):
            fallback = self.user_means.get(user_id, 3.0)
            if verbose:
                print(f"  → All normalized ratings are 0, returning user mean: {fallback:.2f}")
            return fallback
        
        # Final prediction
        weighted = np.dot(weights, ratings) / np.sum(np.abs(weights))
        user_mean = self.user_means.get(user_id, 3.0)
        prediction = float(user_mean + weighted)
        
        if verbose:
            print(f"  → Weighted deviation: {weighted:.3f}")
            print(f"  → User mean: {user_mean:.2f}")
            print(f"  → FINAL PREDICTION: {prediction:.2f}")
        
        return prediction

    # ------------------------------------------------------------
    # Recommend for user
    # ------------------------------------------------------------
    def recommend(self, user_id, top_n=10):
        """
        Generate top-N recommendations for a single user
        """
        print(f"\n{'='*60}")
        print(f"[RECOMMEND] Generating recommendations for User {user_id}")
        print(f"{'='*60}")
        
        user_orig = self.raw_um.loc[user_id]
        unrated = user_orig[user_orig.isna()].index
        
        print(f"[DEBUG] User has rated {user_orig.notna().sum()} movies")
        print(f"[DEBUG] Predicting for {len(unrated)} unrated movies...")
        
        preds = []
        for idx, m in enumerate(unrated):
            score = self.predict(user_id, m)
            preds.append((m, score))
            
            # Progress every 10%
            if (idx + 1) % max(1, len(unrated) // 10) == 0:
                print(f"[PROGRESS] {idx+1}/{len(unrated)} predictions made...")
        
        df = pd.DataFrame(preds, columns=["movieId", "score"])
        df = df.merge(self.movies[["movieId", "title"]], on="movieId", how="left")
        
        result = df.sort_values("score", ascending=False).head(top_n)
        
        print(f"\n[RESULT] Top {top_n} recommendations:")
        print(f"  → Score range: [{result['score'].min():.2f}, {result['score'].max():.2f}]")
        
        return result

    # ------------------------------------------------------------
    # Group recommendation
    # ------------------------------------------------------------
    def recommend_group(self, group_users, method="mean_score", top_n=10):
        """
        Generate recommendations for a group of users
        
        Args:
            group_users: List of user IDs
            method: 'mean_score', 'least_misery', or 'most_pleasure'
            top_n: Number of recommendations
        """
        print(f"\n{'='*60}")
        print(f"[GROUP RECOMMEND] Users: {group_users}")
        print(f"[GROUP RECOMMEND] Method: {method}")
        print(f"{'='*60}")
        
        watched_by_group = self.raw_um.loc[group_users].notna().any(axis=0)
        unrated_movies = self.raw_um.columns[~watched_by_group]
        
        print(f"[DEBUG] Group has watched {watched_by_group.sum()} movies collectively")
        print(f"[DEBUG] Predicting for {len(unrated_movies)} unwatched movies...")
        
        group_scores = {}
        
        for idx, movie_id in enumerate(unrated_movies):
            # Get predictions for all users
            scores = [self.predict(u, movie_id) for u in group_users]
            
            # Aggregate
            if method == "mean_score":
                agg = np.mean(scores)
            elif method == "least_misery":
                agg = np.min(scores)
            elif method == "most_pleasure":
                agg = np.max(scores)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            group_scores[movie_id] = agg
            
            # Progress every 10%
            if (idx + 1) % max(1, len(unrated_movies) // 10) == 0:
                print(f"[PROGRESS] {idx+1}/{len(unrated_movies)} movies processed...")
        
        df = pd.DataFrame(group_scores.items(), columns=["movieId", "score"])
        df = df.merge(self.movies[["movieId", "title"]], on="movieId", how="left")
        
        result = df.sort_values("score", ascending=False).head(top_n)
        
        print(f"\n[RESULT] Top {top_n} group recommendations:")
        print(f"  → Score range: [{result['score'].min():.2f}, {result['score'].max():.2f}]")
        
        return result