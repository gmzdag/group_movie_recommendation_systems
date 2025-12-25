# Evaluation Configuration for Paper
# ===================================

# This file contains all evaluation parameters that should be reported in the paper.
# Update these values based on your final evaluation runs.

# ============================================================================
# TIER 1: OFFLINE EVALUATION (Quantitative Metrics)
# ============================================================================

OFFLINE_EVAL_CONFIG = {
    # Data Configuration
    'data_source': 'MovieLens subset',
    'total_ratings': 106669,
    'total_users': 614,
    'total_movies': 12616,
    'ratings_used': 106669,  
    
    # Split Configuration
    'split_method': 'temporal',
    'train_ratio': 0.70,
    'validation_ratio': 0.15,
    'test_ratio': 0.15,
    
    # Group Configuration
    'num_groups': 30,  # Scientific validity (minimum 30 for statistical power)  
    'min_group_size': 2,
    'max_group_size': 4,
    'min_test_ratings_per_user': 5,
    
    # Evaluation Configuration
    'k_values': [5, 10],
    'ground_truth_threshold': 3.5,  
    'ground_truth_min_support': 1,  
    'ground_truth_strategy': 'relaxed',  
    'candidate_pool_size': 3000, 
    
    # Model Configuration
    'models_evaluated': ['h1', 'h2', 'h3'],
    'normalization': 'zscore',
    'item_k': 60,
    'user_k': 20,  # Optimal from UBCF grid search (Cosine K=20 MIN_OVERLAP=5 → NDCG@10=0.3600)
    'hybrid_weight_C': 1.0,  # Optimal from Hybrid 1 group optimization (Test NDCG@10=0.469)     
    
    # Metrics
    'metrics': [
        'NDCG@K',
        'Precision@K', 
        'Recall@K',
        'Diversity (Jaccard distance)',
        'Coverage (catalog)',
        'Temporal Compatibility'
    ]
}

# ============================================================================
# TIER 2: CASE STUDY EVALUATION (Qualitative Analysis)
# ============================================================================

CASE_STUDY_CONFIG = {
    # User Configuration
    'num_real_users': 3,
    'user_ids': 'z1, z2, z3',  # Anonymized for paper
    
    # Watchlist Data
    'total_watchlist_entries': 205,
    'avg_watchlist_size': 68.3,
    'watchlist_overlap': 'To be calculated',
    
    # Evaluation Type
    'evaluation_type': 'qualitative case study',
    'focus': [
        'Watchlist integration',
        'Explanation quality',
        'Signal source transparency',
        'Real-world applicability'
    ],
    
    # Output
    'top_k': 10,
    'models_demonstrated': ['h1', 'h2', 'h3']
}

if __name__ == "__main__":
    print("="*80)
    print("EVALUATION CONFIGURATION FOR PAPER")
    print("="*80)
    
    print("\n### TIER 1: OFFLINE EVALUATION")
    print("-" * 80)
    for key, value in OFFLINE_EVAL_CONFIG.items():
        print(f"{key:30s}: {value}")
    
    print("\n### TIER 2: CASE STUDY")
    print("-" * 80)
    for key, value in CASE_STUDY_CONFIG.items():
        print(f"{key:30s}: {value}")
    
    print("\n" + USAGE)
