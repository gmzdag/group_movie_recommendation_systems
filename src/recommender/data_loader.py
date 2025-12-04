"""
Data loading utilities for the mixed-hybrid group recommender system.
This module loads movies, ratings, optional watchlists, and prepares 
user–movie matrices and group-level utilities.
"""

import pandas as pd
import numpy as np

MOVIES_PATH = "data/movies_tmdb.csv"
RATINGS_PATH = "data/ratings.csv"
WATCHLIST_PATH = "data/watchlists.csv"


# ------------------------------------------------------
# Load Movies
# ------------------------------------------------------
def load_movies():
    movies = pd.read_csv(MOVIES_PATH)

    # Fill missing text fields
    text_cols = [
        "genres", "Director", "Actors", "Overview",
        "Production_Countries", "Production_Companies", "Keywords"
    ]
    for col in text_cols:
        movies[col] = movies[col].fillna("").astype(str)

    # Numeric fields
    num_cols = ["Vote_Average", "Vote_Count", "Budget", "Revenue", "Runtime"]
    for col in num_cols:
        movies[col] = pd.to_numeric(movies[col], errors="coerce").fillna(0)

    return movies


# ------------------------------------------------------
# Load Ratings
# ------------------------------------------------------
def load_ratings():
    ratings = pd.read_csv(RATINGS_PATH)
    ratings["rating"] = pd.to_numeric(ratings["rating"], errors="coerce").fillna(0)
    return ratings


# ------------------------------------------------------
# Load Optional Watchlists
# ------------------------------------------------------
def load_watchlists():
    try:
        watch = pd.read_csv(WATCHLIST_PATH)
        return watch[["userId", "movieId"]]
    except:
        return pd.DataFrame(columns=["userId", "movieId"])


# ------------------------------------------------------
# Build User–Movie Matrix (for CF models)
# ------------------------------------------------------
def build_user_movie_matrix(ratings):
    return ratings.pivot_table(
        index="userId",
        columns="movieId",
        values="rating",
        aggfunc="mean"
    ).fillna(0)


# ------------------------------------------------------
# Unseen Movies for Group
# ------------------------------------------------------
def get_unseen_movies_for_group(group_user_ids, ratings, movies):
    group_ratings = ratings[ratings["userId"].isin(group_user_ids)]
    seen = set(group_ratings["movieId"])
    all_movies = set(movies["movieId"])
    return sorted(list(all_movies - seen))


# ------------------------------------------------------
# Group Rating Matrix
# ------------------------------------------------------
def get_group_rating_matrix(group_user_ids, ratings):
    subset = ratings[ratings["userId"].isin(group_user_ids)]
    return subset.pivot_table(
        index="userId",
        columns="movieId",
        values="rating",
        aggfunc="mean"
    ).fillna(0)


# ------------------------------------------------------
# Group Watchlist
# ------------------------------------------------------
def get_group_watchlist(group_user_ids, watchlists):
    subset = watchlists[watchlists["userId"].isin(group_user_ids)]
    return sorted(subset["movieId"].unique())


# ------------------------------------------------------
# High-level loader for all models
# ------------------------------------------------------
def load_all_data():
    movies = load_movies()
    ratings = load_ratings()
    watchlists = load_watchlists()
    user_movie_matrix = build_user_movie_matrix(ratings)
    return movies, ratings, watchlists, user_movie_matrix
