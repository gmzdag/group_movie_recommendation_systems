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
        """
        Create synthetic test groups from test users.
        
        SCIENTIFIC APPROACH:
        - Use ALL eligible users from test set (not just specific IDs)
        - Minimum 5 ratings per user for statistical validity
        - Random group formation with fixed seed for reproducibility
        
        Args:
            num_groups: Number of groups to create
            min_size: Minimum group size
            max_size: Maximum group size
            specific_user_ids: IGNORED - we use all eligible users for generalizability
            
        Returns:
            List of user groups
        """
        # Get users with sufficient test ratings (minimum 5 for validity)
        user_counts = self.test_df.groupby('userId').size()
        eligible_users = user_counts[user_counts >= 5].index.tolist()
        
        print(f"[INFO] Found {len(eligible_users)} eligible users in test set (>=5 ratings)")
        
        if len(eligible_users) < min_size:
            print(f"[WARNING] Not enough eligible users ({len(eligible_users)})")
            return []
        
        # Random group formation with fixed seed
        groups = []
        np.random.seed(42)
        
        for _ in range(num_groups):
            current_group_size = min(len(eligible_users), np.random.randint(min_size, max_size + 1))
            group = list(np.random.choice(eligible_users, size=current_group_size, replace=False))
            groups.append(group)
        
        print(f"[INFO] Created {len(groups)} groups (sizes: {[len(g) for g in groups[:5]]}...)")
        return groups
    
    def _evaluate_model_on_groups(self, model_key: str, groups: List[List[int]], k: int = 10) -> float:
        """
        Evaluate a model on test groups using NEGATIVE SAMPLING methodology.
        
        METHODOLOGY (Academic Standard - He et al., 2017):
        - Ground Truth: Union of relevant items (rating >= 3.5) from group members
        - Candidates: Relevant items + 100 randomly sampled negative items
        - Metric: NDCG@K with full ranking
        
        Args:
            model_key: Model key ('h1', 'h2', or 'h3')
            groups: List of user groups
            k: Top-K for NDCG calculation
            
        Returns:
            Mean NDCG@K score across all groups
        """
        from sklearn.metrics import ndcg_score
        
        model = self.models[model_key]
        ndcg_scores = []
        
        # Get all available movies for negative sampling
        all_movies_set = set(self.train_df['movieId'].unique())
        rng = np.random.RandomState(42)  # Fixed seed for reproducibility
        
        for group in groups:
            # 1. GROUND TRUTH (Union Strategy)
            group_test = self.test_df[self.test_df['userId'].isin(group)]
            if group_test.empty:
                continue
            
            # Union: Any member rating >= threshold makes item relevant
            threshold = self.config.get('ground_truth_threshold', 3.5)
            relevant_items = group_test[group_test['rating'] >= threshold]['movieId'].unique().tolist()
            
            if not relevant_items:
                continue
            
            # 2. NEGATIVE SAMPLING (100 Negatives)
            # Get watched items in training (to exclude from negatives)
            watched_train = set()
            for uid in group:
                u_rows = self.train_df[self.train_df['userId'] == uid]
                watched_train.update(u_rows['movieId'].tolist())
            
            # Negatives pool: All movies - Watched in train - Relevant in test
            negatives_pool = list(all_movies_set - watched_train - set(relevant_items))
            
            # Sample 100 negatives
            if len(negatives_pool) > 100:
                negatives = list(rng.choice(negatives_pool, size=100, replace=False))
            else:
                negatives = negatives_pool
            
            # Candidates = Relevant + Negatives
            candidates = relevant_items + negatives
            
            if len(candidates) < k:
                continue
            
            # 3. GET RECOMMENDATIONS (Rank ALL candidates)
            try:
                recs = model.recommend_for_group(group, candidates, top_k=len(candidates))
                if not recs:
                    continue
                
                # 4. CALCULATE NDCG@K
                # Prepare arrays for sklearn
                y_true = []
                y_score = []
                
                for rec in recs:
                    mid = rec['movie_id']
                    score = rec['score']
                    
                    # Binary relevance: 1 if in ground truth, 0 otherwise
                    relevance = 1.0 if mid in relevant_items else 0.0
                    
                    y_true.append(relevance)
                    y_score.append(score)
                
                # Calculate NDCG@K
                if sum(y_true) > 0:  # At least one relevant item
                    ndcg = ndcg_score([y_true], [y_score], k=k)
                    ndcg_scores.append(ndcg)
                    
            except Exception as e:
                print(f"[WARNING] Evaluation failed for {model_key} on group: {e}")
                continue
        
        return np.mean(ndcg_scores) if ndcg_scores else 0.0
    
    def calculate_weights(self, num_groups: int = 30) -> Dict[str, float]:
        """
        Calculate optimal model weights based on NDCG performance.
        
        SCIENTIFIC METHODOLOGY:
        - TRAINING: Models trained on ALL training data
        - EVALUATION: 30 random groups from test set (statistical validity)
        - METRIC: NDCG@10 with Negative Sampling (100 negatives)
        - GROUND TRUTH: Union strategy (any member liking = relevant)
        
        Args:
            num_groups: Number of test groups (default: 30 for statistical power)
            
        Returns:
            Dictionary with weights for h1, h2, h3
        """
        print("\n[WEIGHT OPTIMIZATION] Calculating dynamic model weights...")
        print("[INFO] Using scientifically validated methodology:")
        print("  - Negative Sampling: 100 negatives per group")
        print("  - Ground Truth: Union strategy (rating >= 3.5)")
        print("  - Metric: NDCG@10")
        print(f"  - Groups: {num_groups} (minimum 30 for statistical validity)")
        
        # Create evaluation groups from ALL eligible test users
        groups = self._create_test_groups(num_groups, 2, 5, specific_user_ids=None)
        
        if not groups:
            print("[WARNING] Could not create valid test groups. Falling back to defaults.")
            return {'h1': 0.6, 'h2': 0.3, 'h3': 0.1}
        
        print(f"[INFO] Evaluating models on {len(groups)} test groups...")
        
        # Evaluate all models
        s1 = self._evaluate_model_on_groups('h1', groups, k=10)
        s2 = self._evaluate_model_on_groups('h2', groups, k=10)
        s3 = self._evaluate_model_on_groups('h3', groups, k=10)
        
        print(f"   H1 (Hybrid IBCF+CBF) NDCG@10:  {s1:.4f}")
        print(f"   H2 (Hybrid UBCF+CBF) NDCG@10:  {s2:.4f}")
        print(f"   H3 (Watchlist) NDCG@10:        {s3:.4f}")
        
        # Calculate weights using performance-based distribution
        total_score = s1 + s2 + s3
        
        if total_score < 0.01:
            # If everything failed, use safe defaults
            w1, w2, w3 = 0.6, 0.3, 0.1
            print("[WARNING] All models failed. Using default weights.")
        else:
            w1 = s1 / total_score
            w2 = s2 / total_score
            w3 = s3 / total_score
        
        # Enforce minimums (safety nets) - at least 5% each
        w1 = max(w1, 0.05)
        w2 = max(w2, 0.05)
        w3 = max(w3, 0.05)
        
        # Re-normalize
        total = w1 + w2 + w3
        w1, w2, w3 = w1/total, w2/total, w3/total
        
        weights = {'h1': round(w1, 2), 'h2': round(w2, 2), 'h3': round(w3, 2)}
        print(f"   Calculated Weights: {weights}")
        return weights
    
    def calculate_model_performance(self, num_groups: int = 20, k: int = 10) -> Dict[str, Dict]:
        """
        Calculate individual model performance metrics for multi-model selection.
        
        This method evaluates each model independently and ranks them by NDCG@K.
        Used for quota-based selection in the multi-model recommendation system.
        
        Args:
            num_groups: Number of test groups to create
            k: Top-K for evaluation
            
        Returns:
            Dictionary with performance metrics and rankings:
            {
                'h1': {'ndcg@10': 0.45, 'precision@10': 0.32, 'rank': 1},
                'h2': {'ndcg@10': 0.38, 'precision@10': 0.28, 'rank': 2},
                'h3': {'ndcg@10': 0.25, 'precision@10': 0.18, 'rank': 3}
            }
        """
        from sklearn.metrics import ndcg_score
        
        print(f"\n[MODEL PERFORMANCE] Evaluating individual model performance...")
        
        # Load target users
        target_users = self._load_target_users()
        
        # Create evaluation groups
        groups = self._create_test_groups(num_groups, 2, 5, specific_user_ids=target_users)
        
        if not groups:
            print("[WARNING] Could not create valid test groups. Returning default performance.")
            return {
                'h1': {'ndcg@10': 0.0, 'precision@10': 0.0, 'rank': 1},
                'h2': {'ndcg@10': 0.0, 'precision@10': 0.0, 'rank': 2},
                'h3': {'ndcg@10': 0.0, 'precision@10': 0.0, 'rank': 3}
            }
        
        print(f"[INFO] Evaluating on {len(groups)} target-user groups...")
        
        # Evaluate each model
        performance = {}
        
        for model_key in ['h1', 'h2', 'h3']:
            model = self.models[model_key]
            ndcg_scores = []
            precision_scores = []
            
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
                    
                    # Calculate NDCG
                    true_relevance = [ground_truth.get(mid, 0.0) for mid in rec_ids]
                    if sum(true_relevance) > 0:
                        ndcg = ndcg_score([true_relevance], [rec_scores], k=min(k, len(rec_ids)))
                        ndcg_scores.append(ndcg)
                    
                    # Calculate Precision
                    hits = sum(1 for mid in rec_ids if mid in ground_truth)
                    precision = hits / k
                    precision_scores.append(precision)
                    
                except Exception as e:
                    print(f"[WARNING] Evaluation failed for {model_key}: {e}")
                    continue
            
            # Store metrics
            performance[model_key] = {
                f'ndcg@{k}': round(np.mean(ndcg_scores), 4) if ndcg_scores else 0.0,
                f'precision@{k}': round(np.mean(precision_scores), 4) if precision_scores else 0.0
            }
            
            print(f"   {model_key.upper()}: NDCG@{k}={performance[model_key][f'ndcg@{k}']:.4f}, "
                  f"Precision@{k}={performance[model_key][f'precision@{k}']:.4f}")
        
        # Rank models by NDCG
        sorted_models = sorted(
            performance.items(),
            key=lambda x: x[1][f'ndcg@{k}'],
            reverse=True
        )
        
        for rank, (model_key, metrics) in enumerate(sorted_models, 1):
            performance[model_key]['rank'] = rank
        
        print(f"\n[RANKING] Model ranks: ", end="")
        for model_key in ['h1', 'h2', 'h3']:
            print(f"{model_key.upper()}=#{performance[model_key]['rank']} ", end="")
        print()
        
        return performance


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


def calculate_model_performance_metrics(config: Dict, save_path: str = None) -> Dict[str, Dict]:
    """
    Calculate and save individual model performance metrics.
    
    This function evaluates each model independently and saves performance
    metrics for use in multi-model selection systems.
    
    Args:
        config: Configuration dictionary with evaluation parameters
        save_path: Optional path to save performance JSON (default: data/cache/model_performance.json)
        
    Returns:
        Dictionary with model performance metrics and rankings
    """
    import json
    
    # Load data
    print("[1/3] Loading data...")
    ratings = load_ratings().sort_values('timestamp').tail(config.get('ratings_used', 100000))
    movies = load_movies()
    watchlists = load_watchlists()
    
    print(f"   Using {len(ratings)} ratings")
    
    # Split data
    print("[2/3] Creating temporal splits...")
    train, val, test = temporal_train_validation_test_split(ratings, 0.8, 0.1)
    
    # Calculate performance
    print("[3/3] Calculating model performance...")
    calculator = WeightCalculator(train, val, test, movies, watchlists, config)
    performance = calculator.calculate_model_performance(
        num_groups=config.get('num_groups', 20),
        k=10
    )
    
    # Save to file
    if save_path is None:
        cache_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "data", "cache"
        )
        os.makedirs(cache_dir, exist_ok=True)
        save_path = os.path.join(cache_dir, "model_performance.json")
    
    with open(save_path, 'w') as f:
        json.dump(performance, f, indent=4)
    
    print(f"\n✅ Model performance saved to: {save_path}")
    
    return performance
