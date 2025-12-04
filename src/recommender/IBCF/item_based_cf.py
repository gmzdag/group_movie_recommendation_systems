"""
Item-Based Collaborative Filtering Model
----------------------------------------
Predicts user ratings using item–item similarity and neighbors.
"""

import numpy as np
import pandas as pd


class ItemBasedCF:
    def __init__(self, raw_um, norm_um, item_neighbors, movies, top_k=60):
        """
        raw_um:    U × M matrix with NaN for missing (raw ratings)
        norm_um:   U × M centered rating matrix (mean-centered)
        item_neighbors: dict {movieId: {neighborMovieId: similarity}}
        movies:    movie metadata
        """
        self.raw_um = raw_um
        self.norm_um = norm_um
        self.item_neighbors = item_neighbors
        self.movies = movies
        self.top_k = top_k

    # ------------------------------------------------------------
    # Predict a single rating
    # ------------------------------------------------------------
    def predict(self, user_id: int, movie_id: int) -> float:

        # movie not in model
        if movie_id not in self.item_neighbors:
            rated = self.raw_um.loc[user_id].dropna()
            return rated.mean() if len(rated) else 3.0

        user_orig = self.raw_um.loc[user_id]
        user_norm = self.norm_um.loc[user_id]

        neighbors = self.item_neighbors[movie_id]

        rated_neighbors = [
            m for m in neighbors.keys()
            if pd.notna(user_orig.get(m))
        ]

        if len(rated_neighbors) == 0:
            rated = user_orig.dropna()
            return rated.mean() if len(rated) else 3.0

        weights = np.array([neighbors[m] for m in rated_neighbors])
        ratings = user_norm.loc[rated_neighbors].values

        if np.all(ratings == 0):
            rated = user_orig.dropna()
            return rated.mean() if len(rated) else 3.0

        weighted_sum = np.dot(weights, ratings) / np.sum(np.abs(weights))

        # reconstruct final rating from user's mean
        rated = user_orig.dropna()
        user_mean = rated.mean() if len(rated) else 3.0

        return float(user_mean + weighted_sum)

    # ------------------------------------------------------------
    # Recommend top-N for a user
    # ------------------------------------------------------------
    def recommend(self, user_id: int, top_n=10):
        user_orig = self.raw_um.loc[user_id]
        user_norm = self.norm_um.loc[user_id]

        unrated = user_orig[user_orig.isna()].index

        preds = []
        for movie_id in unrated:
            score = self.predict(user_id, movie_id)
            preds.append((movie_id, score))

        df = pd.DataFrame(preds, columns=["movieId", "score"])
        df = df.merge(self.movies[["movieId", "title"]], on="movieId", how="left")

        return df.sort_values("score", ascending=False).head(top_n)
