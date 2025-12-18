"""
Data Loading Module for Mixed-Hybrid Group Recommender System
-------------------------------------------------------------
This loader provides unified access to:
- Movies
- Ratings
- Watchlists
- User–Movie matrices (CF NaN vs Dense Zero)
- Group utilities

Supports:
- Item-Based CF  → uses NaN matrix
- User-Based CF  → uses NaN matrix (because Pearson/Shrinkage)
- Content-Based Filtering → uses zero-filled matrix
- Mixed Hybrid Models
- Watchlist-based scoring
"""

import pandas as pd
import numpy as np


import os

# ------------------------------------------------------
# File paths
# ------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")

MOVIES_PATH = os.path.join(DATA_DIR, "movies_tmdb.csv")
RATINGS_PATH = os.path.join(DATA_DIR, "ratings.csv")

WATCHLIST_PATH = os.path.join(DATA_DIR, "watchlist.csv")


# ------------------------------------------------------
# Load Movies
# ------------------------------------------------------
def load_movies():
    movies = pd.read_csv(MOVIES_PATH)

    text_cols = [
        "genres", "Director", "Actors", "Overview",
        "Production_Countries", "Production_Companies", "Keywords"
    ]
    for col in text_cols:
        movies[col] = movies[col].fillna("").astype(str)
        
    # Extract Year from Title (e.g., "(1995)")
    movies['year'] = movies['title'].str.extract(r'\((\d{4})\)').astype(float)
    movies['year'] = movies['year'].fillna(0).astype(int)

    num_cols = ["Vote_Average", "Vote_Count", "Budget", "Revenue", "Runtime"]
    for col in num_cols:
        movies[col] = pd.to_numeric(movies[col], errors="coerce").fillna(0)

    return movies


# ------------------------------------------------------
# Load Ratings
# ------------------------------------------------------
def load_ratings():
    ratings = pd.read_csv(RATINGS_PATH)
    ratings["rating"] = pd.to_numeric(ratings["rating"], errors="coerce")
    ratings = ratings.dropna(subset=["rating"])  # remove broken rows
    return ratings


# ------------------------------------------------------
# Load Watchlists
# ------------------------------------------------------
def load_watchlists():
    try:
        watch = pd.read_csv(WATCHLIST_PATH)
        return watch[["userId", "movieId"]]
    except:
        return pd.DataFrame(columns=["userId", "movieId"])


# ------------------------------------------------------
# Matrix for CF (NaN missing)
# ------------------------------------------------------
def build_cf_matrix(ratings):
    """
    CF için kullanılan matris:
    - Missing ratings = NaN
    - Pearson correlation, Shrinkage, Adjusted Cosine için zorunlu
    """
    return ratings.pivot_table(
        index="userId",
        columns="movieId",
        values="rating",
        aggfunc="mean"
    )   # NA korunur


# ------------------------------------------------------
# Dense matrix for CB or group utilities
# ------------------------------------------------------
def build_dense_matrix(ratings):
    """
    Content-based ve group ops için zero-filled matris.
    """
    return ratings.pivot_table(
        index="userId",
        columns="movieId",
        values="rating",
        aggfunc="mean"
    ).fillna(0)


# ------------------------------------------------------
# Group utilities
# ------------------------------------------------------
def get_unseen_movies_for_group(group_user_ids, ratings, movies):
    group_ratings = ratings[ratings["userId"].isin(group_user_ids)]
    seen = set(group_ratings["movieId"])
    all_movies = set(movies["movieId"])
    return sorted(all_movies - seen)


def get_group_rating_matrix(group_user_ids, ratings):
    return ratings[ratings["userId"].isin(group_user_ids)].pivot_table(
        index="userId",
        columns="movieId",
        values="rating",
        aggfunc="mean"
    ).fillna(0)


def get_group_watchlist(group_user_ids, watchlists):
    return sorted(
        watchlists[watchlists["userId"].isin(group_user_ids)]["movieId"].unique()
    )



SPLITS_DIR = os.path.join(DATA_DIR, "splits")

# ------------------------------------------------------
# Load Splits
# ------------------------------------------------------
def load_train_valid_test_splits():
    """
    Loads fixed temporal splits from data/splits/
    Returns: train, validation, test (dataframes)
    """
    if not os.path.exists(SPLITS_DIR):
        raise FileNotFoundError(f"Splits directory not found at {SPLITS_DIR}. Please run create_temporal_splits.py first.")
        
    train = pd.read_csv(os.path.join(SPLITS_DIR, "train.csv"))
    valid = pd.read_csv(os.path.join(SPLITS_DIR, "validation.csv"))
    test = pd.read_csv(os.path.join(SPLITS_DIR, "test.csv"))
    
    return train, valid, test


# ------------------------------------------------------
# High-level unified loader
# ------------------------------------------------------
def load_all_data():
    movies = load_movies()
    ratings = load_ratings()
    watchlists = load_watchlists()

    R_cf = build_cf_matrix(ratings)       # UserCF + ItemCF
    R_dense = build_dense_matrix(ratings) # CB + group ops

    return movies, ratings, watchlists, R_cf, R_dense
