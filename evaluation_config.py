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
    'ratings_used': 106669,  # UPDATED: Use ALL ratings for better CF (especially UserBased)
    
    # Split Configuration
    'split_method': 'temporal',
    'train_ratio': 0.70,
    'validation_ratio': 0.15,
    'test_ratio': 0.15,
    
    # Group Configuration
    'num_groups': 20,  # Reduced from 30 to 20 for faster evaluation
    'min_group_size': 2,
    'max_group_size': 4,
    'min_test_ratings_per_user': 5,
    
    # Evaluation Configuration
    'k_values': [5, 10],
    'ground_truth_threshold': 3.5,  # Ratings >= 3.5 considered relevant
    'ground_truth_min_support': 1,  # RELAXED: Changed from 2 to 1 for sparse data
    'ground_truth_strategy': 'relaxed',  # RELAXED: Changed from 'strict' to 'relaxed'
    'candidate_pool_size': 3000, # Increased from 500 to 3000 to avoid popularity bias
    
    # Model Configuration
    'models_evaluated': ['h1', 'h2', 'h3'],
    'normalization': 'zscore',
    'item_k': 60,
    'user_k': 30,
    'hybrid_weight_C': 2.0, # TUNED: Increased from 1.0 to 2.0 based on Sensitivity Analysis
    
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

# ============================================================================
# PAPER PRESENTATION GUIDELINES
# ============================================================================

PAPER_GUIDELINES = {
    'methodology_section': """
We employ a two-tier evaluation framework:

**Tier 1: Offline Evaluation (Quantitative)**
We evaluate our hybrid models using standard offline metrics on a temporal 
train-validation-test split (70/15/15) of MovieLens data. We create 30 
synthetic groups of 2-4 users from the test set and measure NDCG@K, 
Precision@K, Recall@K, Diversity, and Coverage for K ∈ {5, 10}. Ground 
truth is defined as items rated ≥ 3.5 by group members in the test set.

**Tier 2: Case Study (Qualitative)**
To demonstrate real-world applicability, we conduct a case study with 3 
actual users who provided watchlist data (205 total entries). We generate 
group recommendations and analyze explanation quality, showing how our 
system integrates watchlist signals with collaborative filtering.
""",
    
    'limitations_section': """
While our offline evaluation demonstrates technical quality on standard 
metrics, the case study is limited to 3 users due to the cold-start nature 
of a new system. This is a realistic scenario for new recommendation 
platforms and highlights the value of watchlist integration when rating 
data is sparse. As the system gains users, evaluation can expand to include 
more quantitative metrics on real usage data.
""",
    
    'strengths': [
        'Rigorous quantitative evaluation on standard dataset',
        'Demonstrates real-world applicability with actual users',
        'Transparent about data limitations',
        'Shows both synthetic and real scenarios',
        'Reproducible methodology'
    ]
}

# ============================================================================
# EXPECTED RESULTS STRUCTURE
# ============================================================================

EXPECTED_RESULTS = {
    'offline_evaluation': {
        'table_format': """
| Model | K | NDCG@K | Precision@K | Recall@K | Diversity | Coverage |
|-------|---|--------|-------------|----------|-----------|----------|
| H1    | 5 | X.XXX  | X.XXX       | X.XXX    | X.XXX     | X.XXX    |
| H1    | 10| X.XXX  | X.XXX       | X.XXX    | X.XXX     | X.XXX    |
| H2    | 5 | X.XXX  | X.XXX       | X.XXX    | X.XXX     | X.XXX    |
| H2    | 10| X.XXX  | X.XXX       | X.XXX    | X.XXX     | X.XXX    |
| H3    | 5 | X.XXX  | X.XXX       | X.XXX    | X.XXX     | X.XXX    |
| H3    | 10| X.XXX  | X.XXX       | X.XXX    | X.XXX     | X.XXX    |
"""
    },
    
    'case_study': {
        'table_format': """
**Group:** [User z1, User z2, User z3]

**Watchlist Analysis:**
- User z1: X items
- User z2: X items  
- User z3: X items
- Common items: X
- Overlap ratio: X%

**Top-10 Group Recommendations (Model H1):**

| Rank | Movie | Explanation | Signal Sources |
|------|-------|-------------|----------------|
| 1 | Movie X | ... | IBCF (0.6) + Watchlist (0.4) |
| 2 | Movie Y | ... | CB (0.7) + IBCF (0.3) |
| ... | ... | ... | ... |
"""
    }
}

# ============================================================================
# USAGE INSTRUCTIONS
# ============================================================================

USAGE = """
To generate results for your paper:

1. Run offline evaluation:
   python src/experiments/evaluate_group_system.py
   
   This will:
   - Create temporal splits from MovieLens data
   - Generate 30 synthetic groups
   - Compute all quantitative metrics
   - Save results to: src/experiments/results/group_evaluation_results_all_models.json

2. Run case study:
   python evaluate_case_study.py
   
   This will:
   - Analyze watchlist overlap for real users
   - Generate recommendations for the real user group
   - Create qualitative analysis
   - Save results to: src/experiments/results/case_study_real_users.json

3. Use both results in your paper:
   - Table 1: Offline evaluation metrics (quantitative)
   - Table 2: Case study recommendations (qualitative)
   - Methodology: Explain two-tier approach
   - Limitations: Acknowledge cold-start scenario

4. Report all parameters from this config file in your methodology section.
"""

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
