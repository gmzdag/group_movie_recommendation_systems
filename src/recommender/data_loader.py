"""
Data Loading Module for Mixed-Hybrid Group Recommender System
-------------------------------------------------------------
This loader provides unified access to:
- Movies
- Ratings
- Watchlists
- User–Movie matrices (NaN vs zero-filled)
- Group utilities

It is optimized to support:
- Item-Based CF (uses NaN matrix)
- User-Based CF (uses zero-fill matrix)
- Content-Based Filtering
- Watchlist-based models
- Mixed Hybrid models
"""

import pandas as pd
import numpy as np

# -------------------------------------------------------------------
# File paths (modify easily if needed)
# -------------------------------------------------------------------
MOVIES_PATH = "data/movies_tmdb.csv"
RATINGS_PATH = "data/ratings.csv"
WATCHLIST_PATH = "data/watchlists.csv"


# ------------------------------------------------------
# Load Movies
# ------------------------------------------------------
def load_movies():
    """Load movie metadata and ensure clean text/numeric fields."""
    movies = pd.read_csv(MOVIES_PATH)

    text_cols = [
        "genres", "Director", "Actors", "Overview",
        "Production_Countries", "Production_Companies", "Keywords"
    ]
    for col in text_cols:
        movies[col] = movies[col].fillna("").astype(str)

    num_cols = ["Vote_Average", "Vote_Count", "Budget", "Revenue", "Runtime"]
    for col in num_cols:
        movies[col] = pd.to_numeric(movies[col], errors="coerce").fillna(0)

    return movies


# ------------------------------------------------------
# Load Ratings
# ------------------------------------------------------
def load_ratings():
    """Load MovieLens + Letterboxd-merged ratings."""
    ratings = pd.read_csv(RATINGS_PATH)
    ratings["rating"] = pd.to_numeric(ratings["rating"], errors="coerce")
    ratings = ratings.dropna(subset=["rating"])  # No fake 0s
    return ratings


# ------------------------------------------------------
# Load Watchlists
# ------------------------------------------------------
def load_watchlists():
    """Loads user watchlists if available."""
    try:
        watch = pd.read_csv(WATCHLIST_PATH)
        return watch[["userId", "movieId"]]
    except:
        return pd.DataFrame(columns=["userId", "movieId"])


# ------------------------------------------------------
# User–Movie Matrices
# ------------------------------------------------------
def build_user_movie_matrix_nan(ratings):
    """
    Build U×M matrix with NaN for missing values.
    Required for Item-Based CF.
    """
    return ratings.pivot_table(
        index="userId",
        columns="movieId",
        values="rating",
        aggfunc="mean"
    )  # keep NaN!


def build_user_movie_matrix_zero(ratings):
    """
    Build U×M matrix with 0 for missing values.
    Useful for User-Based CF using cosine similarity.
    """
    return ratings.pivot_table(
        index="userId",
        columns="movieId",
        values="rating",
        aggfunc="mean"
    ).fillna(0)


# ------------------------------------------------------
# Group Utilities
# ------------------------------------------------------
def get_unseen_movies_for_group(group_user_ids, ratings, movies):
    """
    Returns movies that none of the group members have watched.
    Useful in group recommendations.
    """
    group_ratings = ratings[ratings["userId"].isin(group_user_ids)]
    seen = set(group_ratings["movieId"])
    all_movies = set(movies["movieId"])
    return sorted(list(all_movies - seen))


def get_group_rating_matrix(group_user_ids, ratings):
    """
    Returns U_group × M matrix for group-based analysis.
    """
    subset = ratings[ratings["userId"].isin(group_user_ids)]
    return subset.pivot_table(
        index="userId",
        columns="movieId",
        values="rating",
        aggfunc="mean"
    ).fillna(0)


def get_group_watchlist(group_user_ids, watchlists):
    """Return watchlist movies for the group."""
    subset = watchlists[watchlists["userId"].isin(group_user_ids)]
    return sorted(subset["movieId"].unique())


# ------------------------------------------------------
# High-Level Loader (All Data)
# ------------------------------------------------------
def load_all_data():
    """
    Loads everything:
    - movies
    - ratings
    - watchlists
    - NaN-based user–movie matrix (for ItemCF)
    - Zero-fill matrix (for UserCF)
    """
    movies = load_movies()
    ratings = load_ratings()
    watchlists = load_watchlists()

    um_nan = build_user_movie_matrix_nan(ratings)
    um_zero = build_user_movie_matrix_zero(ratings)

    return movies, ratings, watchlists, um_nan, um_zero
