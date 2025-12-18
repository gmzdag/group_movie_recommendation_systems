"""
Group Movie Recommendation System - Complete Demo
--------------------------------------------------
Demonstrates the complete group recommendation pipeline:
1. Group recommendations with explanations
2. Structured 3-section output (Top-10, Watchlist, Themes)
3. Temporal preference filtering

For individual model testing, use:
- demo_hybrid_model_1.py
- demo_hybrid_model_2.py
- demo_group_watchlist.py
"""

import os
import sys
import json

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.model_utils import quick_setup
from src.pipeline.structured_output_generator import StructuredOutputGenerator


def demo_group_recommendations(models, group_users=[618, 623]):
    """Test basic group recommendations."""
    print("\n" + "=" * 80)
    print("DEMO 1: BASIC GROUP RECOMMENDATIONS")
    print("=" * 80)
    
    print(f"\nGroup Users: {group_users}")
    
    # Get candidates
    print("\n[1/2] Generating candidates...")
    
    cf_matrix = models['cf_matrix']
    watched = set()
    for uid in group_users:
        if uid in cf_matrix.index:
            user_watched = cf_matrix.loc[uid].dropna().index.tolist()
            watched.update(user_watched)
            print(f"   User {uid}: {len(user_watched)} movies watched")
    
    # Use popular movies as candidates
    popular = cf_matrix.count().sort_values(ascending=False).head(200).index.tolist()
    candidates = [m for m in popular if m not in watched][:50]
    
    print(f"   Total candidates: {len(candidates)}")
    
    # Get recommendations
    print("\n[2/2] Getting group recommendations...")
    recs = models['h1'].recommend_for_group(group_users, candidates, top_k=5)
    
    print("\n--- Top 5 Group Recommendations ---\n")
    
    movies = models['movies']
    for i, rec in enumerate(recs, 1):
        mid = rec['movie_id']
        score = rec['score']
        group_expl = rec['group_explanation']
        
        # Get title
        movie_row = movies[movies['movieId'] == mid]
        title = movie_row['title'].values[0] if len(movie_row) > 0 else f"Movie {mid}"
        
        print(f"#{i} {title}")
        print(f"   Score: {score:.3f}")
        print(f"   Group Reason: {group_expl}")
        
        print(f"\n   Individual Explanations:")
        for uid, expl in rec['explanations'].items():
            primary = expl.get('primary_reason', 'N/A')
            confidence = expl.get('confidence_level', 'N/A')
            print(f"      User {uid} ({confidence}): {primary}")
        
        print()
    
    print("=" * 80)


def demo_structured_output(models, group_users=[618, 623]):
    """Test structured 3-section output generation."""
    print("\n" + "=" * 80)
    print("DEMO 2: STRUCTURED OUTPUT (3 SECTIONS)")
    print("=" * 80)
    
    print(f"\nGroup Users: {group_users}")
    
    # Initialize generator
    generator = StructuredOutputGenerator(
        hybrid_model_1=models['h1'],
        hybrid_model_2=models['h2'],
        hybrid_model_3=models['h3'],
        movies_df=models['movies'],
        watchlist_df=models['watchlists'],
        cf_matrix=models['cf_matrix'],
        ratings_df=models['ratings'],
        enable_temporal_filtering=True
    )
    
    # Generate output
    print("\nGenerating structured output...")
    output = generator.generate_three_section_output(group_users)
    
    # Display Section A
    print("\n" + "-" * 80)
    print("SECTION A: TOP-10 RANKED GROUP RECOMMENDATIONS")
    print("-" * 80)
    
    section_a = output['section_a_top_recommendations']
    
    if not section_a:
        print("\nNo recommendations in Section A.")
    else:
        for i, rec in enumerate(section_a[:5], 1):  # Show top 5
            print(f"\n#{i} {rec['title']}")
            print(f"   Score: {rec['group_score']}")
            print(f"   Source: {rec['signal_source']}")
            print(f"   Group Explanation: {rec['group_explanation']}")
            
            # Show individual explanations
            print(f"\n   Individual Explanations:")
            for uid, expl in rec['user_explanations'].items():
                primary = expl.get('primary_reason', 'N/A')
                confidence = expl.get('confidence_level', 'N/A')
                print(f"      User {uid} ({confidence}): {primary}")
    
    # Display Section B
    print("\n" + "-" * 80)
    print("SECTION B: COMMON WATCHLIST")
    print("-" * 80)
    
    section_b = output['section_b_common_watchlist']
    
    if not section_b:
        print("\nNo common watchlist items.")
    else:
        for rec in section_b[:3]:  # Show top 3
            print(f"\n• {rec['title']}")
            print(f"   Users: {rec['users']}")
            print(f"   {rec['explanation']}")
    
    # Display Section C
    print("\n" + "-" * 80)
    print("SECTION C: SHARED INTEREST THEMES")
    print("-" * 80)
    
    section_c = output['section_c_shared_interests']
    
    if not section_c:
        print("\nNo shared themes detected.")
    else:
        for i, theme in enumerate(section_c[:2], 1):  # Show top 2
            print(f"\n[THEME {i}] {theme['theme_title']}")
            print(f"   Type: {theme['theme_type']}")
            print(f"   Basis: {theme['explanation_basis']}")
            print(f"\n   Movies:")
            for movie in theme['recommended_movies'][:3]:
                print(f"      • {movie['title']}")
    
    print("\n" + "=" * 80)
    
    return output


def demo_temporal_analysis(models, group_users=[618, 623]):
    """Test temporal preference analysis."""
    print("\n" + "=" * 80)
    print("DEMO 3: TEMPORAL PREFERENCE ANALYSIS")
    print("=" * 80)
    
    from src.recommender.temporal_preference_analyzer import TemporalPreferenceAnalyzer
    
    analyzer = TemporalPreferenceAnalyzer(models['ratings'], models['movies'])
    
    print(f"\nAnalyzing temporal preferences for group: {group_users}\n")
    
    for uid in group_users:
        profile = analyzer.get_user_temporal_profile(uid)
        
        print(f"User {uid}:")
        print(f"   Preference Type: {profile['preference_type']}")
        print(f"   Average Movie Age: {profile['avg_movie_age']:.1f} years")
        print(f"   Recency Score: {profile['recency_score']:.2f} (0=classic, 1=recent)")
        print(f"   Acceptable Year Range: {profile['min_acceptable_year']}-{profile['max_acceptable_year']}")
        print(f"   Total Ratings: {profile['total_ratings']}")
        print()
    
    # Group profile
    group_profile = analyzer.get_group_temporal_profile(group_users)
    print("Group Temporal Profile:")
    print(f"   Preference Type: {group_profile['preference_type']}")
    print(f"   Average Recency: {group_profile['recency_score']:.2f}")
    print(f"   Acceptable Year Range: {group_profile['min_acceptable_year']}-{group_profile['max_acceptable_year']}")
    
    print("\n" + "=" * 80)


def main():
    """Run all group recommendation demos."""
    print("=" * 80)
    print("GROUP MOVIE RECOMMENDATION SYSTEM - COMPLETE DEMO")
    print("=" * 80)
    
    # Quick setup
    print("\n[SETUP] Initializing models...")
    models = quick_setup(
        recent_only=True,
        recent_count=50000,
        normalization='zscore',
        item_k=20,
        user_k=30,
        C=1.0
    )
    
    # Test group
    group_users = [618, 623]
    
    # Run demos
    demo_group_recommendations(models, group_users=group_users)
    output = demo_structured_output(models, group_users=group_users)
    demo_temporal_analysis(models, group_users=group_users)
    
    # Save output
    print("\n" + "=" * 80)
    print("SAVING OUTPUT")
    print("=" * 80)
    
    output_file = os.path.join(
        os.path.dirname(__file__), 
        "..", 
        "group_recommendations_output.json"
    )
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Output saved to: {output_file}")
    print("\n" + "=" * 80)
    print("✅ DEMO COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    import pandas as pd
    main()
