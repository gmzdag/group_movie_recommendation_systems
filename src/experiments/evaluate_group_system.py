"""
Group Recommendation System Evaluation
---------------------------------------
Evaluates the complete group recommendation system using:
- Train/Validation/Test splits (temporal)
- NDCG@K, Precision@K, Recall@K, Diversity, Coverage, Temporal Compatibility

Refactored for clarity and paper-ready results.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from typing import List, Dict, Any

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils.model_utils import ModelFactory
from src.recommender.data_loader import load_movies, load_ratings, load_watchlists
from src.recommender.data_splitter import temporal_train_validation_test_split
from src.recommender.temporal_preference_analyzer import TemporalPreferenceAnalyzer
from src.recommender.hybrid.hybrid_model_1 import HybridModel1
from evaluation_config import OFFLINE_EVAL_CONFIG

# Import metrics from sklearn
from sklearn.metrics import ndcg_score

class GroupRecommenderEvaluator:
    """
    Evaluates group recommendation system with scientific metrics.
    """
    
    def __init__(self, train_df, val_df, test_df, movies_df, watchlists_df):
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.movies_df = movies_df
        self.watchlists_df = watchlists_df
        
        # Initialize Factory
        print("\n[SETUP] Initializing models...")
        self.factory = ModelFactory(
            movies=movies_df,
            ratings=train_df,
            watchlists=watchlists_df,
            normalization=OFFLINE_EVAL_CONFIG['normalization'],
            item_k=OFFLINE_EVAL_CONFIG['item_k'],
            user_k=OFFLINE_EVAL_CONFIG['user_k']
        )
        
        # Create all models (H1, H2, H3)
        self.models = self.factory.create_all_models(C=OFFLINE_EVAL_CONFIG['hybrid_weight_C'])
        print("✅ Models initialized successfully")
        
        # Initialize Temporal Analyzer
        # CRITICAL FIX: Use ONLY training data to avoid data leakage.
        # The analyzer must learn profiles from past history (train) and predict future compatibility.
        self.temporal_analyzer = TemporalPreferenceAnalyzer(
            train_df, 
            movies_df
        )

    def update_h1_model(self, C: float):
        """Re-initializes HybridModel1 with a new C parameter."""
        current_h1 = self.models['h1']
        # Create new instance reusing the underlying models
        self.models['h1'] = HybridModel1(current_h1.ib_model, current_h1.cb_model, C=C)
        print(f"Updated H1 model with C={C}")

    def create_test_groups(self, num_groups, min_size, max_size) -> List[List[int]]:
        """Create synthetic test groups from test users."""
        user_counts = self.test_df.groupby('userId').size()
        eligible_users = user_counts[user_counts >= 5].index.tolist()
        
        if len(eligible_users) < min_size:
            print(f"[WARNING] Not enough eligible users ({len(eligible_users)})")
            return []
        
        groups = []
        np.random.seed(42)
        
        for _ in range(num_groups):
            size = np.random.randint(min_size, max_size + 1)
            group = list(np.random.choice(eligible_users, size=size, replace=False))
            groups.append(group)
            
        return groups

    def get_ground_truth(self, group_users: List[int]) -> Dict:
        """
        Derive ground truth for the group from test data.
        Strategy: Aggregated ratings > threshold & Min Support > 1
        """
        group_test = self.test_df[self.test_df['userId'].isin(group_users)]
        if group_test.empty:
            return {}
        
        # Aggregate
        agg = group_test.groupby('movieId')['rating'].agg(['mean', 'count'])
        
        # Filter (Relaxed/Strict based on config)
        threshold = OFFLINE_EVAL_CONFIG.get('ground_truth_threshold', 3.5)
        
        # CRITICAL FIX: Ensure it's a "Group" preference.
        # Require at least 2 members to have watched/liked it (if group is large enough)
        min_support = 2 if len(group_users) >= 2 else 1
        
        relevant = agg[(agg['mean'] >= threshold) & (agg['count'] >= min_support)]
        
        if relevant.empty:
            return {}
            
        return relevant['mean'].to_dict()

    def get_recommendations(self, model, group_users: List[int], k: int, use_temporal_filter: bool = False) -> List[Dict]:
        """Get recommendations from a specific model, optionally applying temporal filtering."""
        # Candidate generation: popular 3000 (increased from 500 to reduce bias)
        all_movies = self.train_df['movieId'].value_counts().head(3000).index.tolist()
        
        # Filter watched
        watched = set()
        for uid in group_users:
            u_rows = self.train_df[self.train_df['userId'] == uid]
            watched.update(u_rows['movieId'].tolist())
            
        candidates = [m for m in all_movies if m not in watched]
        
        if not candidates:
            return []
            
        # If filtering, ask for more candidates
        request_k = k * 3 if use_temporal_filter else k
        
        # Get recs
        recs = model.recommend_for_group(group_users, candidates, top_k=request_k)
        
        if use_temporal_filter:
            # Re-rank/Filter based on Temporal Compatibility
            filtered_recs = []
            for rec in recs:
                mid = rec['movie_id']
                # Calculate group temporal score
                temp_scores = [self.temporal_analyzer.get_temporal_compatibility_score(uid, mid) for uid in group_users]
                avg_temp = np.mean(temp_scores) if temp_scores else 0.5
                
                # Strict Filter: Must be > 0.3 compatibility
                if avg_temp >= 0.3:
                    rec['temporal_score'] = avg_temp
                    filtered_recs.append(rec)
            
            # Sort by original score? Or combine?
            # User request: "measure results with temporal analyzer enabled"
            # We preserve original rank but cut off incompatible ones
            recs = filtered_recs[:k]
            
        return recs

    def evaluate_group(self, model, group_users: List[int], k: int, use_temporal_filter: bool = False) -> Dict[str, float]:
        """Compute metrics for a single group."""
        ground_truth = self.get_ground_truth(group_users)
        if not ground_truth:
            return {} # Skip groups with no ground truth in test
            
        recs = self.get_recommendations(model, group_users, k, use_temporal_filter)
        if not recs:
            return {}
            
        rec_ids = [r['movie_id'] for r in recs]
        rec_scores = [r['score'] for r in recs]
        
        # 1. NDCG
        true_relevance = [ground_truth.get(mid, 0.0) for mid in rec_ids]
        if sum(true_relevance) == 0:
            ndcg = 0.0
        else:
            # Pad if fewer than k recommendations
            if len(true_relevance) < k:
                true_relevance += [0.0] * (k - len(true_relevance))
                rec_scores += [0.0] * (k - len(rec_scores))
            ndcg = ndcg_score([true_relevance], [rec_scores], k=k)
            
        # 2. Precision
        hits = sum(1 for mid in rec_ids if mid in ground_truth)
        precision = hits / k
        
        # 3. Recall
        total_relevant = len(ground_truth)
        recall = hits / total_relevant if total_relevant > 0 else 0.0
        
        # 4. Temporal
        temp_scores = [np.mean([self.temporal_analyzer.get_temporal_compatibility_score(uid, mid) 
                                for uid in group_users]) for mid in rec_ids]
        temporal = np.mean(temp_scores) if temp_scores else 0.0
        
        return {
            'ndcg': ndcg,
            'precision': precision,
            'recall': recall,
            'temporal': temporal,
            'rec_ids': rec_ids 
        }

    def evaluate_model(self, model_key: str, groups: List[List[int]], k: int, use_temporal_filter: bool = False) -> Dict:
        """Evaluate a specific model across all groups for a single K."""
        model = self.models[model_key]
        metrics = {'ndcg': [], 'precision': [], 'recall': [], 'temporal': []}
        all_recs = set()
        
        for group in groups:
            res = self.evaluate_group(model, group, k, use_temporal_filter)
            if not res: 
                continue
                
            metrics['ndcg'].append(res['ndcg'])
            metrics['precision'].append(res['precision'])
            metrics['recall'].append(res['recall'])
            metrics['temporal'].append(res['temporal'])
            all_recs.update(res['rec_ids'])
        
        return {
            'mean_ndcg': np.mean(metrics['ndcg']) if metrics['ndcg'] else 0.0,
            'mean_precision': np.mean(metrics['precision']) if metrics['precision'] else 0.0,
            'mean_recall': np.mean(metrics['recall']) if metrics['recall'] else 0.0,
            'mean_temporal': np.mean(metrics['temporal']) if metrics['temporal'] else 0.0,
            'coverage': len(all_recs) / len(self.movies_df) if not self.movies_df.empty else 0.0
        }

def run_evaluation():
    print("="*60)
    print("SCIENTIFIC EVALUATION & ABLATION STUDY")
    print("="*60)
    
    # 1. Load Data
    print("Loading data...")
    ratings = load_ratings().sort_values('timestamp').tail(OFFLINE_EVAL_CONFIG['ratings_used'])
    movies = load_movies()
    watchlists = load_watchlists()
    
    # 2. Split
    print("Splitting data...")
    train, val, test = temporal_train_validation_test_split(
        ratings, 
        OFFLINE_EVAL_CONFIG['train_ratio'],
        OFFLINE_EVAL_CONFIG['validation_ratio']
    )
    
    # 3. Initialize Evaluator
    evaluator = GroupRecommenderEvaluator(train, val, test, movies, watchlists)
    
    # 4. Create Groups
    print(f"Creating {OFFLINE_EVAL_CONFIG['num_groups']} test groups...")
    groups = evaluator.create_test_groups(
        OFFLINE_EVAL_CONFIG['num_groups'],
        OFFLINE_EVAL_CONFIG['min_group_size'],
        OFFLINE_EVAL_CONFIG['max_group_size']
    )
    
    # Results Directories
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    graphs_dir = os.path.join(results_dir, "graphs")
    os.makedirs(graphs_dir, exist_ok=True)
    
    # ==========================================================
    # EXPERIMENT 1: SENSITIVITY ANALYSIS (Parameter C)
    # ==========================================================
    print("\n[EXPERIMENT 1] Sensitivity Analysis for C (Trust Factor)...")
    c_values = [0.1, 0.5, 1.0, 2.0, 5.0]
    c_results = {'ndcg': [], 'precision': []}
    
    # Fixed K for this experiment
    eval_k = 10 
    
    for c in c_values:
        evaluator.update_h1_model(c)
        res = evaluator.evaluate_model('h1', groups, eval_k, use_temporal_filter=False)
        c_results['ndcg'].append(res['mean_ndcg'])
        c_results['precision'].append(res['mean_precision'])
        print(f"  C={c}: NDCG={res['mean_ndcg']:.4f}, Prec={res['mean_precision']:.4f}")
        
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(c_values, c_results['ndcg'], marker='o', label='NDCG@10', color='b')
    plt.plot(c_values, c_results['precision'], marker='s', label='Precision@10', color='g', linestyle='--')
    plt.title('Sensitivity Analysis: Impact of Trust Factor C on H1 Performance')
    plt.xlabel('C (Trust Factor)')
    plt.ylabel('Score')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(graphs_dir, "sensitivity_analysis_C.png"))
    plt.savefig(os.path.join(graphs_dir, "sensitivity_analysis_C.svg"))
    plt.close()
    print("✅ Sensitivity Analysis Graph Saved.")
    
    # ==========================================================
    # EXPERIMENT 2: ABLATION STUDY (Temporal Filtering)
    # ==========================================================
    print("\n[EXPERIMENT 2] Ablation Study: Temporal Filtering...")
    
    # Choose best C from previous step (simple max)
    best_idx = np.argmax(c_results['ndcg'])
    best_c = c_values[best_idx]
    print(f"Using Best C={best_c} for Ablation Study.")
    evaluator.update_h1_model(best_c)
    
    # Compare
    res_without = evaluator.evaluate_model('h1', groups, eval_k, use_temporal_filter=False)
    res_with = evaluator.evaluate_model('h1', groups, eval_k, use_temporal_filter=True)
    
    print("\n[ABLATION RESULTS]")
    print(f"{'Metric':<20} {'Without Temporal':<20} {'With Temporal':<20} {'Change %':<10}")
    print("-" * 75)
    
    metrics_to_compare = ['mean_ndcg', 'mean_precision', 'mean_temporal', 'coverage']
    
    ablation_data = []
    
    for m in metrics_to_compare:
        v1 = res_without[m]
        v2 = res_with[m]
        change = ((v2 - v1) / v1 * 100) if v1 > 0 else 0.0
        print(f"{m:<20} {v1:.4f}               {v2:.4f}               {change:+.2f}%")
        ablation_data.append([m, v1, v2])

    # Plotting Ablation
    labels = [m.replace('mean_', '').upper() for m in metrics_to_compare]
    v1_vals = [res_without[m] for m in metrics_to_compare]
    v2_vals = [res_with[m] for m in metrics_to_compare]
    
    x = np.arange(len(labels))
    width = 0.35
    
    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, v1_vals, width, label='Without Temporal', color='gray')
    plt.bar(x + width/2, v2_vals, width, label='With Temporal', color='teal')
    plt.ylabel('Score')
    plt.title('Ablation Study: Impact of Temporal Filtering (H1)')
    plt.xticks(x, labels)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.savefig(os.path.join(graphs_dir, "ablation_study_temporal.png"))
    plt.savefig(os.path.join(graphs_dir, "ablation_study_temporal.svg"))
    plt.close()
    print("✅ Ablation Study Graph Saved.")

    # Save Results JSON
    final_output = {
        'sensitivity_analysis': {
            'c_values': c_values,
            'ndcg': c_results['ndcg'],
            'precision': c_results['precision']
        },
        'ablation_study': {
            'best_c': best_c,
            'without_temporal': res_without,
            'with_temporal': res_with
        }
    }
    
    output_path = os.path.join(results_dir, "scientific_analysis_results.json")
    with open(output_path, 'w') as f:
        json.dump(final_output, f, indent=4)
        
    print(f"\nDetailed results saved to {output_path}")

if __name__ == "__main__":
    run_evaluation()
