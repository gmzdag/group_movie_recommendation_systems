"""
Group Recommendation System Evaluation
---------------------------------------
Evaluates the complete group recommendation system using:
- Train/Validation/Test splits (temporal)
- NDCG@K (Normalized Discounted Cumulative Gain)
- Precision@K
- Recall@K
- Coverage
- Diversity
- Temporal compatibility metrics
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import time

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils.model_utils import ModelFactory
from src.recommender.data_loader import load_movies, load_ratings, load_watchlists
from src.recommender.data_splitter import temporal_train_validation_test_split
from src.recommender.temporal_preference_analyzer import TemporalPreferenceAnalyzer
from sklearn.metrics import ndcg_score


class GroupRecommenderEvaluator:
    """Evaluates group recommendation system with scientific metrics."""
    
    def __init__(self, train_df, val_df, test_df, movies_df, watchlists_df):
        """
        Initialize evaluator.
        
        Args:
            train_df: Training ratings
            val_df: Validation ratings
            test_df: Test ratings
            movies_df: Movie metadata
            watchlists_df: User watchlists
        """
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.movies_df = movies_df
        self.watchlists_df = watchlists_df
        
        # Initialize models on training data
        print("\n[SETUP] Initializing models on training data...")
        self.factory = ModelFactory(
            movies=movies_df,
            ratings=train_df,
            watchlists=watchlists_df,
            normalization='zscore',
            item_k=20,
            user_k=30
        )
        
        self.models = self.factory.create_all_models(C=1.0)
        print("✅ Models initialized")
        
        # Initialize temporal analyzer
        self.temporal_analyzer = TemporalPreferenceAnalyzer(
            pd.concat([train_df, val_df, test_df]), 
            movies_df
        )
    
    def create_test_groups(self, min_group_size=2, max_group_size=4, num_groups=50) -> List[List[int]]:
        """
        Create test groups from users in test set.
        
        Args:
            min_group_size: Minimum group size
            max_group_size: Maximum group size
            num_groups: Number of groups to create
            
        Returns:
            List of user ID lists
        """
        # Get users with sufficient test ratings
        user_counts = self.test_df.groupby('userId').size()
        eligible_users = user_counts[user_counts >= 5].index.tolist()
        
        if len(eligible_users) < min_group_size:
            print(f"[WARNING] Not enough eligible users ({len(eligible_users)})")
            return []
        
        groups = []
        np.random.seed(42)
        
        for _ in range(num_groups):
            group_size = np.random.randint(min_group_size, max_group_size + 1)
            if len(eligible_users) >= group_size:
                group = list(np.random.choice(eligible_users, size=group_size, replace=False))
                groups.append(group)
        
        return groups
    
    def get_group_ground_truth(self, group_users: List[int], k: int = 10) -> Dict:
        """
        Get ground truth for a group from test set.
        
        RELAXED STRATEGY (Better for sparse data):
        - Aggregate individual preferences (rating >= 3.5, more lenient)
        - Weight by number of members who rated it
        - More realistic for sparse test sets
        
        Returns:
            {
                'relevant_movies': [movie_ids],
                'ratings': {movie_id: avg_rating}
            }
        """
        # Get test ratings for group members
        group_test = self.test_df[self.test_df['userId'].isin(group_users)]
        
        if group_test.empty:
            return {'relevant_movies': [], 'ratings': {}}
        
        # Aggregate ratings per movie
        movie_agg = group_test.groupby('movieId').agg({
            'rating': ['mean', 'count']
        })
        
        # Filter: avg rating >= 3.5 (more lenient than 4.0)
        relevant = movie_agg[movie_agg[('rating', 'mean')] >= 3.5]
        
        if relevant.empty:
            return {'relevant_movies': [], 'ratings': {}}
        
        # Sort by: count (popularity in group) then rating
        relevant = relevant.sort_values(
            [('rating', 'count'), ('rating', 'mean')], 
            ascending=False
        )
        
        ratings_dict = dict(relevant[('rating', 'mean')])
        relevant_movies = list(relevant.index)[:k*3]
        
        return {
            'relevant_movies': relevant_movies,
            'ratings': ratings_dict
        }
    
    def get_group_recommendations(self, group_users: List[int], k: int = 10) -> List[int]:
        """
        Get top-K recommendations for a group.
        
        Args:
            group_users: List of user IDs
            k: Number of recommendations
            
        Returns:
            List of movie IDs
        """
        # Get candidates (movies not watched by any group member in train)
        cf_matrix = self.models['cf_matrix']
        watched = set()
        
        for uid in group_users:
            if uid in cf_matrix.index:
                user_watched = cf_matrix.loc[uid].dropna().index.tolist()
                watched.update(user_watched)
        
        # Use popular movies as candidates
        all_movies = self.train_df['movieId'].value_counts().head(500).index.tolist()
        candidates = [m for m in all_movies if m not in watched][:200]
        
        if not candidates:
            return []
        
        # Get recommendations using Hybrid Model 1
        try:
            recs = self.models['h1'].recommend_for_group(group_users, candidates, top_k=k)
            return [rec['movie_id'] for rec in recs]
        except Exception as e:
            print(f"[ERROR] Recommendation failed: {e}")
            return []
    
    def calculate_ndcg_at_k(self, group_users: List[int], k: int = 10) -> float:
        """
        Calculate NDCG@K for a group.
        
        NDCG measures ranking quality considering relevance scores.
        FIXED: Now uses actual model scores instead of ranks.
        """
        # Get ground truth
        ground_truth = self.get_group_ground_truth(group_users, k=k*2)
        relevant_movies = ground_truth['relevant_movies']
        ratings_dict = ground_truth['ratings']
        
        if not relevant_movies:
            return 0.0
        
        # Get candidates
        cf_matrix = self.models['cf_matrix']
        watched = set()
        for uid in group_users:
            if uid in cf_matrix.index:
                user_watched = cf_matrix.loc[uid].dropna().index.tolist()
                watched.update(user_watched)
        
        all_movies = self.train_df['movieId'].value_counts().head(500).index.tolist()
        candidates = [m for m in all_movies if m not in watched][:200]
        
        if not candidates:
            return 0.0
        
        # Get recommendations WITH SCORES
        try:
            recs = self.models['h1'].recommend_for_group(group_users, candidates, top_k=k)
        except Exception as e:
            print(f"[ERROR] Recommendation failed: {e}")
            return 0.0
        
        if not recs:
            return 0.0
        
        # Use ACTUAL MODEL SCORES (not ranks!)
        recommended_ids = [rec['movie_id'] for rec in recs]
        true_relevance = [ratings_dict.get(mid, 0.0) for mid in recommended_ids]
        pred_relevance = [rec['score'] for rec in recs]  # ACTUAL SCORES
        
        # Calculate NDCG
        try:
            ndcg = ndcg_score([true_relevance], [pred_relevance], k=k)
            return ndcg
        except:
            return 0.0
    
    def calculate_precision_at_k(self, group_users: List[int], k: int = 10) -> float:
        """
        Calculate Precision@K for a group.
        
        Precision@K = (# relevant items in top-K) / K
        """
        ground_truth = self.get_group_ground_truth(group_users, k=k*2)
        relevant_movies = set(ground_truth['relevant_movies'])
        
        if not relevant_movies:
            return 0.0
        
        recommended = self.get_group_recommendations(group_users, k=k)
        
        if not recommended:
            return 0.0
        
        hits = len(set(recommended[:k]) & relevant_movies)
        precision = hits / k
        
        return precision
    
    def calculate_recall_at_k(self, group_users: List[int], k: int = 10) -> float:
        """
        Calculate Recall@K for a group.
        
        Recall@K = (# relevant items in top-K) / (total # relevant items)
        """
        ground_truth = self.get_group_ground_truth(group_users, k=k*2)
        relevant_movies = set(ground_truth['relevant_movies'])
        
        if not relevant_movies:
            return 0.0
        
        recommended = self.get_group_recommendations(group_users, k=k)
        
        if not recommended:
            return 0.0
        
        hits = len(set(recommended[:k]) & relevant_movies)
        recall = hits / len(relevant_movies)
        
        return recall
    
    def calculate_coverage(self, groups: List[List[int]], k: int = 10) -> float:
        """
        Calculate catalog coverage.
        
        Coverage = (# unique items recommended) / (total # items)
        """
        all_recommended = set()
        
        for group in groups:
            recs = self.get_group_recommendations(group, k=k)
            all_recommended.update(recs)
        
        total_items = len(self.movies_df)
        coverage = len(all_recommended) / total_items
        
        return coverage
    
    def calculate_diversity(self, group_users: List[int], k: int = 10) -> float:
        """
        Calculate intra-list diversity (average pairwise distance).
        
        Uses genre-based Jaccard distance.
        """
        recommended = self.get_group_recommendations(group_users, k=k)
        
        if len(recommended) < 2:
            return 0.0
        
        # Get genres for each movie
        movie_genres = {}
        for mid in recommended:
            movie_row = self.movies_df[self.movies_df['movieId'] == mid]
            if not movie_row.empty:
                genres = set(str(movie_row.iloc[0]['genres']).split('|'))
                movie_genres[mid] = genres
        
        # Calculate pairwise Jaccard distances
        distances = []
        for i, mid1 in enumerate(recommended):
            for mid2 in recommended[i+1:]:
                if mid1 in movie_genres and mid2 in movie_genres:
                    g1 = movie_genres[mid1]
                    g2 = movie_genres[mid2]
                    
                    if len(g1 | g2) > 0:
                        jaccard_sim = len(g1 & g2) / len(g1 | g2)
                        jaccard_dist = 1 - jaccard_sim
                        distances.append(jaccard_dist)
        
        return np.mean(distances) if distances else 0.0
    
    def calculate_temporal_compatibility(self, group_users: List[int], k: int = 10) -> float:
        """
        Calculate temporal compatibility score.
        
        Measures how well recommendations match group's temporal preferences.
        """
        recommended = self.get_group_recommendations(group_users, k=k)
        
        if not recommended:
            return 0.0
        
        # Get temporal compatibility scores
        scores = []
        for mid in recommended:
            # Average compatibility across group members
            member_scores = []
            for uid in group_users:
                score = self.temporal_analyzer.get_temporal_compatibility_score(uid, mid)
                member_scores.append(score)
            
            avg_score = np.mean(member_scores)
            scores.append(avg_score)
        
        return np.mean(scores)
    
    def evaluate_all_metrics(self, groups: List[List[int]], k: int = 10, model_key: str = 'h1') -> Dict:
        """
        Evaluate all metrics across all groups.
        
        Args:
            groups: List of user ID lists
            k: Number of recommendations
            model_key: Which model to use ('h1', 'h2', 'h3', or 'current')
        
        Returns:
            Dictionary with metric results
        """
        print(f"\n{'='*80}")
        print(f"EVALUATING {len(groups)} GROUPS WITH K={k} (Model: {model_key.upper()})")
        print(f"{'='*80}\n")
        
        # Store original model
        original_model = self.models.get('h1')
        
        # Use specified model
        if model_key != 'h1':
            self.models['h1'] = self.models[model_key]
        
        ndcg_scores = []
        precision_scores = []
        recall_scores = []
        diversity_scores = []
        temporal_scores = []
        
        start_time = time.time()
        
        for i, group in enumerate(groups, 1):
            if i % 10 == 0:
                elapsed = time.time() - start_time
                eta = (elapsed / i) * (len(groups) - i)
                print(f"[PROGRESS] {i}/{len(groups)} groups | Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s")
            
            try:
                ndcg = self.calculate_ndcg_at_k(group, k=k)
                precision = self.calculate_precision_at_k(group, k=k)
                recall = self.calculate_recall_at_k(group, k=k)
                diversity = self.calculate_diversity(group, k=k)
                temporal = self.calculate_temporal_compatibility(group, k=k)
                
                ndcg_scores.append(ndcg)
                precision_scores.append(precision)
                recall_scores.append(recall)
                diversity_scores.append(diversity)
                temporal_scores.append(temporal)
                
            except Exception as e:
                print(f"[ERROR] Group {i} failed: {e}")
                continue
        
        # Calculate coverage
        coverage = self.calculate_coverage(groups, k=k)
        
        elapsed = time.time() - start_time
        
        # Restore original model
        if model_key != 'h1':
            self.models['h1'] = original_model
        
        results = {
            'model': model_key,
            'k': k,
            'num_groups': len(groups),
            'ndcg@k': {
                'mean': np.mean(ndcg_scores),
                'std': np.std(ndcg_scores),
                'min': np.min(ndcg_scores),
                'max': np.max(ndcg_scores)
            },
            'precision@k': {
                'mean': np.mean(precision_scores),
                'std': np.std(precision_scores),
                'min': np.min(precision_scores),
                'max': np.max(precision_scores)
            },
            'recall@k': {
                'mean': np.mean(recall_scores),
                'std': np.std(recall_scores),
                'min': np.min(recall_scores),
                'max': np.max(recall_scores)
            },
            'diversity': {
                'mean': np.mean(diversity_scores),
                'std': np.std(diversity_scores),
                'min': np.min(diversity_scores),
                'max': np.max(diversity_scores)
            },
            'temporal_compatibility': {
                'mean': np.mean(temporal_scores),
                'std': np.std(temporal_scores),
                'min': np.min(temporal_scores),
                'max': np.max(temporal_scores)
            },
            'coverage': coverage,
            'evaluation_time_seconds': elapsed
        }
        
        return results


def print_results(results: Dict):
    """Pretty print evaluation results."""
    print(f"\n{'='*80}")
    print(f"EVALUATION RESULTS (K={results['k']})")
    print(f"{'='*80}\n")
    
    print(f"Number of Groups Evaluated: {results['num_groups']}")
    print(f"Evaluation Time: {results['evaluation_time_seconds']:.2f}s\n")
    
    print(f"{'Metric':<30} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
    print(f"{'-'*80}")
    
    for metric_name in ['ndcg@k', 'precision@k', 'recall@k', 'diversity', 'temporal_compatibility']:
        metric = results[metric_name]
        print(f"{metric_name:<30} {metric['mean']:<12.4f} {metric['std']:<12.4f} "
              f"{metric['min']:<12.4f} {metric['max']:<12.4f}")
    
    print(f"\nCatalog Coverage: {results['coverage']:.4f} ({results['coverage']*100:.2f}%)")
    print(f"{'='*80}\n")


def main():
    """Run complete evaluation."""
    print("="*80)
    print("GROUP RECOMMENDATION SYSTEM - SCIENTIFIC EVALUATION")
    print("="*80)
    
    # Load data
    print("\n[1/5] Loading data...")
    movies = load_movies()
    ratings = load_ratings()
    watchlists = load_watchlists()
    
    # Use recent data for faster evaluation
    ratings = ratings.sort_values('timestamp').tail(100000)
    
    print(f"   Movies: {len(movies)}")
    print(f"   Ratings: {len(ratings)}")
    print(f"   Users: {ratings['userId'].nunique()}")
    
    # Create temporal splits
    print("\n[2/5] Creating temporal train/val/test splits...")
    train_df, val_df, test_df = temporal_train_validation_test_split(
        ratings, 
        train_ratio=0.7, 
        valid_ratio=0.15
    )
    
    print(f"   Train: {len(train_df)} ratings")
    print(f"   Validation: {len(val_df)} ratings")
    print(f"   Test: {len(test_df)} ratings")
    
    # Initialize evaluator
    print("\n[3/5] Initializing evaluator...")
    evaluator = GroupRecommenderEvaluator(train_df, val_df, test_df, movies, watchlists)
    
    # Create test groups
    print("\n[4/5] Creating test groups...")
    groups = evaluator.create_test_groups(
        min_group_size=2,
        max_group_size=4,
        num_groups=30  # Start with 30 groups for faster testing
    )
    
    print(f"   Created {len(groups)} test groups")
    print(f"   Group sizes: {[len(g) for g in groups[:5]]}... (showing first 5)")
    
    # Evaluate ALL HYBRID MODELS
    print("\n[5/5] Running evaluation for ALL hybrid models...")
    
    all_results = {}
    
    for model_name in ['h1', 'h2', 'h3']:
        print(f"\n{'='*80}")
        print(f"EVALUATING MODEL: {model_name.upper()}")
        print(f"{'='*80}")
        
        # Temporarily replace the model in evaluator
        evaluator.models['current'] = evaluator.models[model_name]
        
        # Evaluate for K=5 and K=10
        results_k5 = evaluator.evaluate_all_metrics(groups, k=5, model_key='current')
        results_k10 = evaluator.evaluate_all_metrics(groups, k=10, model_key='current')
        
        all_results[model_name] = {
            'k5': results_k5,
            'k10': results_k10
        }
        
        # Print results
        print_results(results_k5)
        print_results(results_k10)
    
    # Save results
    output_file = os.path.join(
        os.path.dirname(__file__),
        "results",
        "group_evaluation_results_all_models.json"
    )
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_file}")
    
    # Print comparison table
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    print(f"\n{'Model':<10} {'K':<5} {'NDCG':<10} {'Precision':<12} {'Recall':<10} {'Diversity':<10}")
    print("-"*80)
    
    for model_name in ['h1', 'h2', 'h3']:
        for k in [5, 10]:
            key = f'k{k}'
            res = all_results[model_name][key]
            print(f"{model_name.upper():<10} {k:<5} {res['ndcg@k']['mean']:<10.4f} "
                  f"{res['precision@k']['mean']:<12.4f} {res['recall@k']['mean']:<10.4f} "
                  f"{res['diversity']['mean']:<10.4f}")
    
    print("\n" + "="*80)
    print("✅ EVALUATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
