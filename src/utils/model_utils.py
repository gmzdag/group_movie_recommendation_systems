"""
Model Utilities
---------------
Centralized utilities for model initialization, normalization, and common operations.
This module eliminates code duplication across demos, experiments, and main scripts.
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import Tuple, Dict, Optional
import os
import sys

# Add src to path if needed
if 'src' not in sys.path:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.recommender.data_loader import load_movies, load_ratings, load_watchlists, build_cf_matrix
from src.recommender.IBCF.item_based_cf import ItemBasedCF
from src.recommender.IBCF.neighbors_item import compute_item_neighbors
from src.recommender.UBCF.user_based_cf import UserBasedCF
from src.recommender.UBCF.neighbors_user import precompute_all_user_neighbors
from src.recommender.UBCF.similarity_user import pearson_shrink
from src.recommender.CB.content_based import ContentBasedModel
from src.recommender.hybrid.hybrid_model_1 import HybridModel1
from src.recommender.hybrid.hybrid_model_2 import SwitchingHybridRecommender
from src.recommender.hybrid.hybrid_model_3 import WatchlistHybridModel


# ============================================================================
# NORMALIZATION FUNCTIONS
# ============================================================================

def normalize_zscore(matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Apply Z-score normalization to a user-item matrix.
    
    Formula: (rating - user_mean) / user_std
    
    Args:
        matrix: User-item rating matrix (users as rows, items as columns)
        
    Returns:
        Normalized matrix with same shape
    """
    user_means = matrix.mean(axis=1)
    user_stds = matrix.std(axis=1).replace(0, 1)  # Avoid division by zero
    
    normalized = matrix.sub(user_means, axis=0).div(user_stds, axis=0).fillna(0)
    
    return normalized


def normalize_mean_centering(matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Apply mean-centering normalization (subtract user mean).
    
    Args:
        matrix: User-item rating matrix
        
    Returns:
        Mean-centered matrix
    """
    user_means = matrix.mean(axis=1)
    return matrix.sub(user_means, axis=0).fillna(0)


def normalize_min_max(matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Apply min-max normalization per user (scale to 0-1).
    
    Args:
        matrix: User-item rating matrix
        
    Returns:
        Min-max normalized matrix
    """
    user_min = matrix.min(axis=1)
    user_max = matrix.max(axis=1)
    user_range = (user_max - user_min).replace(0, 1)
    
    normalized = matrix.sub(user_min, axis=0).div(user_range, axis=0).fillna(0)
    
    return normalized


# ============================================================================
# DATA LOADING UTILITIES
# ============================================================================

def load_and_prepare_data(
    sample_size: Optional[int] = None,
    recent_only: bool = False,
    recent_count: int = 50000
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and optionally sample the dataset.
    
    Args:
        sample_size: If provided, randomly sample this many ratings
        recent_only: If True, take only the most recent ratings
        recent_count: Number of recent ratings to keep if recent_only=True
        
    Returns:
        Tuple of (movies, ratings, watchlists)
    """
    movies = load_movies()
    ratings = load_ratings()
    watchlists = load_watchlists()
    
    if recent_only:
        ratings = ratings.sort_values('timestamp').tail(recent_count)
    elif sample_size:
        ratings = ratings.sample(n=min(sample_size, len(ratings)), random_state=42)
    
    return movies, ratings, watchlists


# ============================================================================
# MODEL INITIALIZATION
# ============================================================================

class ModelFactory:
    """
    Factory class for initializing recommendation models with standard configurations.
    """
    
    def __init__(
        self,
        movies: pd.DataFrame,
        ratings: pd.DataFrame,
        watchlists: pd.DataFrame,
        normalization: str = 'zscore',
        item_k: int = 20,
        user_k: int = 30
    ):
        """
        Initialize the model factory.
        
        Args:
            movies: Movies DataFrame
            ratings: Ratings DataFrame
            watchlists: Watchlists DataFrame
            normalization: 'zscore', 'mean_center', or 'min_max'
            item_k: Number of item neighbors for IBCF
            user_k: Number of user neighbors for UBCF
        """
        self.movies = movies
        self.ratings = ratings
        self.watchlists = watchlists
        self.normalization = normalization
        self.item_k = item_k
        self.user_k = user_k
        
        # Build matrices
        self.cf_matrix = build_cf_matrix(ratings)
        
        # Apply normalization
        if normalization == 'zscore':
            self.norm_matrix = normalize_zscore(self.cf_matrix)
        elif normalization == 'mean_center':
            self.norm_matrix = normalize_mean_centering(self.cf_matrix)
        elif normalization == 'min_max':
            self.norm_matrix = normalize_min_max(self.cf_matrix)
        else:
            raise ValueError(f"Unknown normalization: {normalization}")
        
        # Compute similarities and neighbors
        self._compute_similarities()
        
        # Calculate statistics for UBCF
        self.user_means = self.cf_matrix.mean(axis=1)
        self.item_means = self.cf_matrix.mean(axis=0)
        self.global_mean = self.cf_matrix.stack().mean()
    
    def _compute_similarities(self):
        """Compute item and user similarities."""
        # Item similarity (cosine on normalized matrix)
        item_sim_matrix = cosine_similarity(self.norm_matrix.fillna(0).T)
        self.item_sim_df = pd.DataFrame(
            item_sim_matrix,
            index=self.norm_matrix.columns,
            columns=self.norm_matrix.columns
        )
        
        # Compute item neighbors
        self.item_neighbors = compute_item_neighbors(self.item_sim_df, K=self.item_k)
        
        # Compute user neighbors
        self.user_neighbors = precompute_all_user_neighbors(
            self.cf_matrix, 
            pearson_shrink, 
            K=self.user_k
        )
    
    def create_ibcf(self, top_k: int = 20) -> ItemBasedCF:
        """Create Item-Based CF model."""
        return ItemBasedCF(
            raw_um=self.cf_matrix,
            norm_um=self.norm_matrix,
            item_neighbors=self.item_neighbors,
            movies=self.movies,
            top_k=top_k
        )
    
    def create_ubcf(self) -> UserBasedCF:
        """Create User-Based CF model."""
        return UserBasedCF(
            R=self.cf_matrix,
            neighbors=self.user_neighbors,
            user_means=self.user_means,
            item_means=self.item_means,
            global_mean=self.global_mean,
            movies=self.movies
        )
    
    def create_cbf(self) -> ContentBasedModel:
        """Create Content-Based Filtering model."""
        return ContentBasedModel(
            movies_df=self.movies,
            ratings_df=self.ratings
        )
    
    def create_hybrid_model_1(self, C: float = 1.0) -> HybridModel1:
        """
        Create Hybrid Model 1 (Dynamic Weighted: IBCF + CBF).
        
        Args:
            C: Confidence hyperparameter
        """
        ibcf = self.create_ibcf()
        cbf = self.create_cbf()
        
        return HybridModel1(ibcf, cbf, C=C)
    
    def create_hybrid_model_2(self) -> SwitchingHybridRecommender:
        """Create Hybrid Model 2 (Switching: UBCF + CBF)."""
        ubcf = self.create_ubcf()
        cbf = self.create_cbf()
        
        return SwitchingHybridRecommender(ubcf, cbf)
    
    def create_hybrid_model_3(self) -> WatchlistHybridModel:
        """Create Hybrid Model 3 (Watchlist-Driven)."""
        cbf = self.create_cbf()
        
        return WatchlistHybridModel(
            movies=self.movies,
            watchlist_df=self.watchlists,
            cbf_model=cbf
        )
    
    def create_all_models(self, C: float = 1.0) -> Dict:
        """
        Create all models at once.
        
        Returns:
            Dict with keys: 'ibcf', 'ubcf', 'cbf', 'h1', 'h2', 'h3'
        """
        ibcf = self.create_ibcf()
        ubcf = self.create_ubcf()
        cbf = self.create_cbf()
        
        h1 = HybridModel1(ibcf, cbf, C=C)
        h2 = SwitchingHybridRecommender(ubcf, cbf)
        h3 = WatchlistHybridModel(self.movies, self.watchlists, cbf)
        
        return {
            'ibcf': ibcf,
            'ubcf': ubcf,
            'cbf': cbf,
            'h1': h1,
            'h2': h2,
            'h3': h3,
            'cf_matrix': self.cf_matrix,
            'norm_matrix': self.norm_matrix
        }


# ============================================================================
# QUICK SETUP FUNCTION
# ============================================================================

def quick_setup(
    sample_size: Optional[int] = None,
    recent_only: bool = False,
    recent_count: int = 50000,
    normalization: str = 'zscore',
    item_k: int = 20,
    user_k: int = 30,
    C: float = 1.0
) -> Dict:
    """
    One-line setup for all models with standard configuration.
    
    Example:
        >>> models = quick_setup(recent_only=True, recent_count=50000)
        >>> h1 = models['h1']
        >>> score = h1.predict(user_id, movie_id)
    
    Args:
        sample_size: Number of ratings to sample (None = all)
        recent_only: Use only recent ratings
        recent_count: Number of recent ratings if recent_only=True
        normalization: Normalization method
        item_k: Item neighbors for IBCF
        user_k: User neighbors for UBCF
        C: Confidence parameter for Hybrid Model 1
        
    Returns:
        Dict with all models and data
    """
    print("=" * 70)
    print("QUICK SETUP: Loading and initializing models...")
    print("=" * 70)
    
    # Load data
    print("\n[1/3] Loading data...")
    movies, ratings, watchlists = load_and_prepare_data(
        sample_size=sample_size,
        recent_only=recent_only,
        recent_count=recent_count
    )
    
    print(f"   Movies: {len(movies)}")
    print(f"   Ratings: {len(ratings)}")
    print(f"   Watchlist entries: {len(watchlists)}")
    
    # Initialize factory
    print(f"\n[2/3] Building matrices and computing similarities...")
    print(f"   Normalization: {normalization}")
    print(f"   Item neighbors (K): {item_k}")
    print(f"   User neighbors (K): {user_k}")
    
    factory = ModelFactory(
        movies=movies,
        ratings=ratings,
        watchlists=watchlists,
        normalization=normalization,
        item_k=item_k,
        user_k=user_k
    )
    
    # Create all models
    print(f"\n[3/3] Initializing all models...")
    models = factory.create_all_models(C=C)
    
    # Add data to output
    models['movies'] = movies
    models['ratings'] = ratings
    models['watchlists'] = watchlists
    models['factory'] = factory
    
    print("\n" + "=" * 70)
    print("✅ SETUP COMPLETE")
    print("=" * 70)
    print("\nAvailable models:")
    print("  - models['ibcf']: Item-Based CF")
    print("  - models['ubcf']: User-Based CF")
    print("  - models['cbf']: Content-Based Filtering")
    print("  - models['h1']: Hybrid Model 1 (Dynamic Weighted)")
    print("  - models['h2']: Hybrid Model 2 (Switching)")
    print("  - models['h3']: Hybrid Model 3 (Watchlist-Driven)")
    print("  - models['cf_matrix']: CF Matrix")
    print("  - models['movies']: Movies DataFrame")
    print("  - models['ratings']: Ratings DataFrame")
    print("  - models['watchlists']: Watchlists DataFrame")
    print("=" * 70 + "\n")
    
    return models


# ============================================================================
# DEBUGGING UTILITIES
# ============================================================================

def print_matrix_stats(matrix: pd.DataFrame, name: str = "Matrix"):
    """Print statistics about a matrix."""
    print(f"\n{name} Statistics:")
    print(f"  Shape: {matrix.shape[0]} users × {matrix.shape[1]} items")
    print(f"  Total values: {matrix.size:,}")
    print(f"  Non-null values: {matrix.count().sum():,}")
    print(f"  Sparsity: {(1 - matrix.count().sum() / matrix.size) * 100:.2f}%")
    print(f"  Mean: {matrix.stack().mean():.4f}")
    print(f"  Std: {matrix.stack().std():.4f}")
    print(f"  Min: {matrix.min().min():.4f}")
    print(f"  Max: {matrix.max().max():.4f}")


def validate_model_setup(models: Dict) -> bool:
    """
    Validate that all models are properly initialized.
    
    Returns:
        True if all checks pass
    """
    required_keys = ['ibcf', 'ubcf', 'cbf', 'h1', 'h2', 'h3', 'cf_matrix']
    
    print("\n" + "=" * 70)
    print("MODEL VALIDATION")
    print("=" * 70)
    
    all_ok = True
    
    for key in required_keys:
        if key in models and models[key] is not None:
            print(f"✅ {key}: OK")
        else:
            print(f"❌ {key}: MISSING")
            all_ok = False
    
    print("=" * 70)
    
    if all_ok:
        print("✅ All models validated successfully")
    else:
        print("❌ Some models are missing or invalid")
    
    print("=" * 70 + "\n")
    
    return all_ok
