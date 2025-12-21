"""
Weight Calculator Module
-------------------------
Centralized module for calculating optimal model weights based on performance metrics.
This module contains all weight calculation logic extracted from various scripts.
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.recommender.data_loader import load_movies, load_ratings, load_watchlists
from src.recommender.data_splitter import temporal_train_validation_test_split
from src.utils.model_utils import ModelFactory


class WeightCalculator:
    """
    Calculates optimal weights for ensemble models based on performance metrics.
    """
    
    def __init__(self, train_df, val_df, test_df, movies_df, watchlists_df, config: Dict):
        """
        Initialize weight calculator.
        
        Args:
            train_df: Training data
            val_df: Validation data
            test_df: Test data
            movies_df: Movies metadata
            watchlists_df: User watchlists
            config: Configuration dictionary with model parameters
        """
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.movies_df = movies_df
        self.watchlists_df = watchlists_df
        self.config = config
        
        # Initialize models
        print("\n[WEIGHT CALCULATOR] Initializing models...")
        self.factory = ModelFactory(
            movies=movies_df,
            ratings=train_df,
            watchlists=watchlists_df,
            normalization=config.get('normalization', 'zscore'),
            item_k=config.get('item_k', 60),
            user_k=config.get('user_k', 30)
        )
        
        self.models = self.factory.create_all_models(C=config.get('hybrid_weight_C', 1.0))
        print("✅ Models initialized successfully")
    
    def _load_target_users(self) -> List[int]:
        """Load target user IDs from data/users.csv"""
        users_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "data", "users.csv"
        )
        try:
            users_df = pd.read_csv(users_path)
            valid_ids = users_df['user_id'].dropna().astype(int).tolist()
            print(f"[INFO] Loaded {len(valid_ids)} target users from users.csv")
            return valid_ids
        except Exception as e:
            print(f"[WARNING] Could not load users.csv: {e}. Using all users.")
            return []
    
    def _create_test_groups(self, num_groups: int, min_size: int, max_size: int, 
                           specific_user_ids: List[int] = None) -> List[List[int]]:
        """Create synthetic test groups from test users."""
        if specific_user_ids:
            eligible_users = [u for u in specific_user_ids if u in self.test_df['userId'].unique()]
            print(f"[INFO] Restricting groups to {len(eligible_users)} specific users found in Test Data.")
        else:
            user_counts = self.test_df.groupby('userId').size()
            eligible_users = user_counts[user_counts >= 5].index.tolist()
        
        if len(eligible_users) < min_size:
            print(f"[WARNING] Not enough eligible users ({len(eligible_users)})")
            return []
        
        if specific_user_ids and len(eligible_users) <= 10:
            import itertools
            groups = []
            for size in range(min_size, min(max_size, len(eligible_users)) + 1):
                combos = list(itertools.combinations(eligible_users, size))
                groups.extend([list(c) for c in combos])
            
            if len(groups) > num_groups:
                import random
                random.seed(42)
                random.shuffle(groups)
                groups = groups[:num_groups]
            print(f"[INFO] Generated {len(groups)} permutations for small user set.")
            return groups

        groups = []
        np.random.seed(42)
        
        for _ in range(num_groups):
            current_group_size = min(len(eligible_users), np.random.randint(min_size, max_size + 1))
            group = list(np.random.choice(eligible_users, size=current_group_size, replace=False))
            groups.append(group)
            
        return groups
    
    def _evaluate_model_on_groups(self, model_key: str, groups: List[List[int]], k: int = 10) -> float:
        """
        Evaluate a model on test groups and return mean NDCG.
        
        Args:
            model_key: Model key ('h1', 'h2', or 'h3')
            groups: List of user groups
            k: Top-K for evaluation
            
        Returns:
            Mean NDCG score
        """
        from sklearn.metrics import ndcg_score
        
        model = self.models[model_key]
        ndcg_scores = []
        
        for group in groups:
            # Get ground truth
            group_test = self.test_df[self.test_df['userId'].isin(group)]
            if group_test.empty:
                continue
            
            agg = group_test.groupby('movieId')['rating'].agg(['mean', 'count'])
            threshold = self.config.get('ground_truth_threshold', 3.5)
            min_support = max(1, int(len(group) * 0.3))
            
            relevant = agg[(agg['mean'] >= threshold) & (agg['count'] >= min_support)]
            if relevant.empty:
                continue
            
            ground_truth = relevant['mean'].to_dict()
            
            # Get recommendations
            all_movies = self.train_df['movieId'].value_counts().head(2000).index.tolist()
            watched = set()
            for uid in group:
                u_rows = self.train_df[self.train_df['userId'] == uid]
                watched.update(u_rows['movieId'].tolist())
            
            candidates = [m for m in all_movies if m not in watched]
            if not candidates:
                continue
            
            try:
                recs = model.recommend_for_group(group, candidates, top_k=k)
                if not recs:
                    continue
                
                rec_ids = [r['movie_id'] for r in recs]
                rec_scores = [r['score'] for r in recs]
                
                true_relevance = [ground_truth.get(mid, 0.0) for mid in rec_ids]
                if sum(true_relevance) > 0:
                    ndcg = ndcg_score([true_relevance], [rec_scores], k=min(k, len(rec_ids)))
                    ndcg_scores.append(ndcg)
            except Exception as e:
                print(f"[WARNING] Evaluation failed for {model_key}: {e}")
                continue
        
        return np.mean(ndcg_scores) if ndcg_scores else 0.0
    
    def calculate_weights(self, num_groups: int = 20) -> Dict[str, float]:
        """
        Calculate optimal model weights based on NDCG performance.
        
        STRATEGY:
        - TRAINING: ALL models (H1, H2, H3) are trained on ALL users for better CF quality
        - EVALUATION: Groups are composed ONLY of target users from users.csv
        - This ensures personalized weight calibration while maintaining model quality
        
        Args:
            num_groups: Number of test groups to create
            
        Returns:
            Dictionary with weights for h1, h2, h3
        """
        print("\n[WEIGHT OPTIMIZATION] Calculating dynamic model weights...")
        print("[INFO] Models are trained on ALL users, but calibration uses TARGET user groups")
        
        # Load target users
        target_users = self._load_target_users()
        
        # Create evaluation groups using ONLY target users
        groups = self._create_test_groups(num_groups, 2, 5, specific_user_ids=target_users)
        
        if not groups:
            print("[WARNING] Could not create valid test groups from target users. Falling back to defaults.")
            return {'h1': 0.6, 'h2': 0.3, 'h3': 0.1}
        
        print(f"[INFO] Calibrating ALL models on {len(groups)} target-user group configurations...")
        
        # Evaluate all models
        s1 = self._evaluate_model_on_groups('h1', groups, k=10)
        s2 = self._evaluate_model_on_groups('h2', groups, k=10)
        s3 = self._evaluate_model_on_groups('h3', groups, k=10)
        
        print(f"   H1 (Hybrid) NDCG:    {s1:.4f}")
        print(f"   H2 (Switching) NDCG: {s2:.4f}")
        print(f"   H3 (Watchlist) NDCG: {s3:.4f}")
        
        # Calculate weights using softmax-like distribution
        total_score = s1 + s2 + s3
        
        if total_score < 0.01:
            # If everything failed, use safe defaults
            w1, w2, w3 = 0.5, 0.3, 0.2
        else:
            w1 = s1 / total_score
            w2 = s2 / total_score
            w3 = s3 / total_score
        
        # Enforce minimums (safety nets)
        w1 = max(w1, 0.1)
        w2 = max(w2, 0.1)
        w3 = max(w3, 0.1)
        
        # Re-normalize
        total = w1 + w2 + w3
        w1, w2, w3 = w1/total, w2/total, w3/total
        
        weights = {'h1': round(w1, 2), 'h2': round(w2, 2), 'h3': round(w3, 2)}
        print(f"   Calculated Weights: {weights}")
        return weights


def calculate_production_weights(config: Dict) -> Dict[str, float]:
    """
    Standalone function to calculate production weights.
    
    Args:
        config: Configuration dictionary with evaluation parameters
        
    Returns:
        Dictionary with model weights
    """
    # Load data
    print("[1/3] Loading data...")
    ratings = load_ratings().sort_values('timestamp').tail(config.get('ratings_used', 100000))
    movies = load_movies()
    watchlists = load_watchlists()
    
    print(f"   Using {len(ratings)} ratings")
    
    # Split data
    print("[2/3] Creating temporal splits...")
    train, val, test = temporal_train_validation_test_split(ratings, 0.8, 0.1)
    
    # Calculate weights
    print("[3/3] Calculating weights...")
    calculator = WeightCalculator(train, val, test, movies, watchlists, config)
    weights = calculator.calculate_weights(num_groups=config.get('num_groups', 20))
    
    return weights
