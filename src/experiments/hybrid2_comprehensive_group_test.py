"""
HYBRID MODEL 2 - COMPREHENSIVE GROUP OPTIMIZATION
=================================================

GOAL: Find OPTIMAL configuration for Hybrid Model 2 in GROUP context

TESTED DIMENSIONS:
1. STRATEGY: Performance-Weighted vs Confidence-Weighted
2. PARAMETERS: Weights (w) or Trust factor (C)
3. AGGREGATION: AVERAGE vs LEAST_MISERY vs HARMONIC_MEAN

OUTPUT: Best overall configuration for group recommendations

SCIENTIFIC VALIDITY:
- 30 validation groups (Amer-Yahia et al., 2009 standard)
- Group NDCG@10 + Fairness metrics
"""

import os
import sys
import numpy as np
import pandas as pd
from functools import partial
from typing import List, Dict, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from recommender.data_loader import load_movies, load_train_valid_test_splits, build_cf_matrix
from recommender.UBCF.user_based_cf import UserBasedCF
from recommender.UBCF.neighbors_user import load_or_compute_neighbors
from recommender.UBCF.similarity_user import cosine_sim
from recommender.CB.content_based import ContentBasedModel
from sklearn.metrics import ndcg_score

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


class ComprehensiveHybrid2:
    """Unified Hybrid 2 with all strategies and aggregations."""
    
    def __init__(self, ubcf, cbf, strategy='performance', w_ubcf=0.05, w_cbf=0.95, C=10.0, aggregation='average'):
        self.ubcf = ubcf
        self.cbf = cbf
        self.strategy = strategy  # 'performance' or 'confidence'
        self.w_ubcf = w_ubcf
        self.w_cbf = w_cbf
        self.C = C
        self.aggregation = aggregation  # 'average', 'least_misery', 'harmonic_mean'
    
    def predict(self, uid, mid):
        """Individual prediction using selected strategy."""
        try:
            ubcf_pred = self.ubcf.predict(uid, mid)
        except:
            ubcf_pred = np.nan
        
        cbf_pred = self.cbf.predict_rating(uid, mid)
        
        # Handle NaNs
        if np.isnan(ubcf_pred) and np.isnan(cbf_pred):
            return self.ubcf.global_mean
        elif np.isnan(ubcf_pred):
            return cbf_pred
        elif np.isnan(cbf_pred):
            return ubcf_pred
        
        # Apply strategy
        if self.strategy == 'performance':
            # Static weighted
            return self.w_ubcf * ubcf_pred + self.w_cbf * cbf_pred
        else:  # confidence
            # Dynamic weighted
            n_neighbors = len(self.ubcf.neighbors.get(uid, {}))
            confidence = n_neighbors / (n_neighbors + self.C)
            return confidence * ubcf_pred + (1 - confidence) * cbf_pred
    
    def recommend_for_group(self, group_users, candidates, top_k=50):
        """Group recommendation with selected aggregation."""
        group_scores = []
        
        for mid in candidates:
            member_scores = []
            
            for uid in group_users:
                try:
                    score = self.predict(uid, mid)
                    if not np.isnan(score):
                        member_scores.append(score)
                except:
                    pass
            
            if not member_scores:
                continue
            
            # Apply aggregation strategy
            if self.aggregation == 'average':
                final_score = np.mean(member_scores)
            elif self.aggregation == 'least_misery':
                final_score = np.min(member_scores)
            elif self.aggregation == 'harmonic_mean':
                if all(s > 0 for s in member_scores):
                    final_score = len(member_scores) / sum(1.0/s for s in member_scores)
                else:
                    final_score = 0.0
            else:
                final_score = np.mean(member_scores)
            
            group_scores.append((mid, final_score, member_scores))
        
        group_scores.sort(key=lambda x: x[1], reverse=True)
        return group_scores[:top_k]


def create_validation_groups(valid_df, num_groups=30):
    """Create validation groups."""
    user_ratings = valid_df.groupby('userId').size()
    eligible_users = user_ratings[user_ratings >= 5].index.tolist()
    
    if len(eligible_users) < 2 * num_groups:
        num_groups = len(eligible_users) // 2
    
    np.random.seed(42)
    np.random.shuffle(eligible_users)
    
    groups = []
    for i in range(num_groups):
        group_size = np.random.randint(2, 5)  # 2-4 members
        start_idx = i * 2
        end_idx = start_idx + group_size
        
        if end_idx > len(eligible_users):
            break
        
        group = eligible_users[start_idx:end_idx]
        groups.append(group)
    
    return groups


def evaluate_model(model, groups, valid_df, R_train, k=10):
    """Evaluate model on groups."""
    group_ndcgs = []
    group_fairness = []
    
    for group_users in groups:
        group_valid = valid_df[valid_df['userId'].isin(group_users)]
        
        if len(group_valid) < k:
            continue
        
        # Candidates
        all_watched = set()
        for uid in group_users:
            if uid in R_train.index:
                all_watched.update(R_train.loc[uid].dropna().index)
        
        all_items = R_train.columns.tolist()
        candidates = [m for m in all_items if m not in all_watched]
        
        if len(candidates) < k:
            continue
        
        # Recommendations
        try:
            recs = model.recommend_for_group(group_users, candidates, top_k=k)
        except:
            continue
        
        if not recs:
            continue
        
        # NDCG
        y_true = []
        y_score = []
        
        for mid, score, member_scores in recs:
            item_ratings = group_valid[group_valid['movieId'] == mid]['rating'].tolist()
            avg_rating = np.mean(item_ratings) if item_ratings else 0.0
            
            y_true.append(avg_rating)
            y_score.append(score)
        
        if sum(y_true) > 0:
            try:
                ndcg = ndcg_score([y_true], [y_score])
                group_ndcgs.append(ndcg)
                
                # Fairness
                all_member_scores = [ms for _, _, ms in recs for ms in ms]
                if all_member_scores:
                    fairness = np.min(all_member_scores) / np.mean(all_member_scores)
                    group_fairness.append(fairness)
            except:
                pass
    
    avg_ndcg = np.mean(group_ndcgs) if group_ndcgs else 0.0
    avg_fairness = np.mean(group_fairness) if group_fairness else 0.0
    
    return avg_ndcg, avg_fairness, len(group_ndcgs)


def main():
    print("="*80)
    print(" HYBRID MODEL 2 - COMPREHENSIVE GROUP OPTIMIZATION")
    print(" Testing: Strategy × Parameters × Aggregation")
    print("="*80)
    
    # Load
    print("\n[1] Loading Data...")
    movies = load_movies()
    train_df, valid_df, test_df = load_train_valid_test_splits()
    print(f"  Train: {len(train_df):,} | Valid: {len(valid_df):,}")
    
    # Build
    print("\n[2] Building Models...")
    R_train = build_cf_matrix(train_df)
    user_means = R_train.mean(axis=1)
    item_means = R_train.mean(axis=0)
    global_mean = train_df["rating"].mean()
    
    print("  UBCF (Cosine, K=50)...")
    neighbors = load_or_compute_neighbors(
        R_train, partial(cosine_sim, MIN_OVERLAP=5), K=50,
        metric="cosine_overlap5_hybrid2_opt"
    )
    ubcf = UserBasedCF(R_train, neighbors, user_means, item_means, global_mean, movies=movies)
    
    print("  CBF...")
    cbf = ContentBasedModel(movies, train_df)
    
    # Groups
    print("\n[3] Creating Groups...")
    groups = create_validation_groups(valid_df, num_groups=30)
    print(f"  Created {len(groups)} groups")
    print(f"  {'✅' if len(groups) >= 30 else '⚠️ '} Groups: {len(groups)}")
    
    # Define test configurations
    configs = []
    
    # Performance-Weighted configurations
    print("\n[4] COMPREHENSIVE TEST")
    print("="*80)
    
    for w_ubcf in [0.0, 0.05, 0.10, 0.15, 0.20]:
        w_cbf = 1.0 - w_ubcf
        for agg in ['average', 'least_misery', 'harmonic_mean']:
            configs.append({
                'strategy': 'performance',
                'w_ubcf': w_ubcf,
                'w_cbf': w_cbf,
                'C': None,
                'aggregation': agg
            })
    
    # Confidence-Weighted configurations
    for C in [5, 10, 20, 30]:
        for agg in ['average', 'least_misery', 'harmonic_mean']:
            configs.append({
                'strategy': 'confidence',
                'w_ubcf': None,
                'w_cbf': None,
                'C': C,
                'aggregation': agg
            })
    
    print(f"\nTotal configurations to test: {len(configs)}")
    print(f"Estimated time: ~{len(configs) * 2} minutes\n")
    
    # Test all configurations
    results = []
    
    for idx, config in enumerate(configs, 1):
        # Create model
        model = ComprehensiveHybrid2(
            ubcf, cbf,
            strategy=config['strategy'],
            w_ubcf=config.get('w_ubcf', 0.05),
            w_cbf=config.get('w_cbf', 0.95),
            C=config.get('C', 10.0),
            aggregation=config['aggregation']
        )
        
        # Describe config
        if config['strategy'] == 'performance':
            param_desc = f"w={config['w_ubcf']:.2f}/{config['w_cbf']:.2f}"
        else:
            param_desc = f"C={config['C']}"
        
        config_desc = f"{config['strategy'].capitalize()}({param_desc})+{config['aggregation'].upper()}"
        
        print(f"[{idx}/{len(configs)}] Testing: {config_desc:<50}", end='')
        
        # Evaluate
        ndcg, fairness, n_groups = evaluate_model(model, groups, valid_df, R_train)
        
        print(f" NDCG={ndcg:.4f}, Fair={fairness:.4f}, N={n_groups}")
        
        # Store
        results.append({
            'strategy': config['strategy'],
            'parameter': param_desc,
            'aggregation': config['aggregation'],
            'w_ubcf': config.get('w_ubcf'),
            'C': config.get('C'),
            'group_ndcg': ndcg,
            'fairness': fairness,
            'composite_score': 0.7 * ndcg + 0.3 * fairness,  # Weighted metric
            'num_groups': n_groups
        })
    
    # Analysis
    print("\n" + "="*80)
    print(" RESULTS")
    print("="*80)
    
    df = pd.DataFrame(results)
    
    # Sort by composite score
    df_sorted = df.sort_values('composite_score', ascending=False)
    
    print("\n🏆 TOP 10 CONFIGURATIONS:")
    print(df_sorted.head(10)[['strategy', 'parameter', 'aggregation', 'group_ndcg', 'fairness', 'composite_score']].to_string(index=False))
    
    # Best overall
    best = df_sorted.iloc[0]
    
    print("\n" + "="*80)
    print(" ✅ OPTIMAL CONFIGURATION FOR GROUP RECOMMENDATIONS")
    print("="*80)
    print(f"\nStrategy: {best['strategy'].upper()}")
    print(f"Parameter: {best['parameter']}")
    print(f"Aggregation: {best['aggregation'].upper()}")
    print(f"\nPerformance:")
    print(f"  Group NDCG@10: {best['group_ndcg']:.4f}")
    print(f"  Fairness: {best['fairness']:.4f}")
    print(f"  Composite Score: {best['composite_score']:.4f}")
    print(f"  Groups Evaluated: {best['num_groups']}")
    
    # Best per dimension
    print("\n" + "="*80)
    print(" BEST PER DIMENSION")
    print("="*80)
    
    print("\nBest Strategy:")
    for strategy in ['performance', 'confidence']:
        best_strategy = df[df['strategy'] == strategy].sort_values('composite_score', ascending=False).iloc[0]
        print(f"  {strategy.capitalize()}: {best_strategy['composite_score']:.4f} ({best_strategy['parameter']}, {best_strategy['aggregation']})")
    
    print("\nBest Aggregation:")
    for agg in ['average', 'least_misery', 'harmonic_mean']:
        best_agg = df[df['aggregation'] == agg].sort_values('composite_score', ascending=False).iloc[0]
        print(f"  {agg.upper()}: {best_agg['composite_score']:.4f} ({best_agg['strategy']}, {best_agg['parameter']})")
    
    # Save
    csv_path = os.path.join(RESULTS_DIR, "hybrid2_comprehensive_group_optimization.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n💾 Saved: {csv_path}")
    
    # Recommendation
    print("\n" + "="*80)
    print(" FINAL RECOMMENDATION")
    print("="*80)
    print(f"\nUse Hybrid Model 2 with:")
    print(f"  - {best['strategy'].upper()} strategy ({best['parameter']})")
    print(f"  - {best['aggregation'].upper()} aggregation")
    print(f"  - Expected Group NDCG@10: {best['group_ndcg']:.4f}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
