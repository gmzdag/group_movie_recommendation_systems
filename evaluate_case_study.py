"""
Case Study Evaluation - Real Users with Watchlist Data
------------------------------------------------------
This script generates a qualitative case study for the paper,
showing how the system works with actual users (z1, z2, z3).

Focus: Explanation quality and watchlist integration
NOT: Quantitative metrics (not enough users for statistical significance)
"""

import os
import sys
import json
import pandas as pd
from typing import List, Dict

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.model_utils import ModelFactory
from src.recommender.data_loader import load_movies, load_ratings, load_watchlists


def analyze_watchlist_overlap(watchlists_df: pd.DataFrame, user_ids: List[int]) -> Dict:
    """Analyze watchlist overlap for a group."""
    group_watchlists = {}
    
    for uid in user_ids:
        user_items = watchlists_df[watchlists_df['userId'] == uid]['movieId'].tolist()
        group_watchlists[uid] = set(user_items)
    
    # Find common items
    if group_watchlists:
        common_items = set.intersection(*group_watchlists.values())
    else:
        common_items = set()
    
    # Find union
    all_items = set.union(*group_watchlists.values()) if group_watchlists else set()
    
    return {
        'individual_sizes': {uid: len(items) for uid, items in group_watchlists.items()},
        'common_items': list(common_items),
        'total_unique_items': len(all_items),
        'overlap_ratio': len(common_items) / len(all_items) if all_items else 0
    }


def get_movie_details(movie_id: int, movies_df: pd.DataFrame) -> Dict:
    """Get movie details for display."""
    movie = movies_df[movies_df['movieId'] == movie_id]
    
    if movie.empty:
        return {'title': f'Unknown Movie {movie_id}', 'genres': 'Unknown'}
    
    movie = movie.iloc[0]
    return {
        'movie_id': movie_id,
        'title': movie['title'],
        'genres': movie['genres'],
        'year': movie.get('year', 'N/A'),
        'overview': movie.get('Overview', '')[:200] + '...' if len(str(movie.get('Overview', ''))) > 200 else movie.get('Overview', '')
    }


def generate_case_study_report(group_users: List[int], top_k: int = 10):
    """Generate a comprehensive case study report for the paper."""
    
    print("="*80)
    print("CASE STUDY: REAL USER GROUP RECOMMENDATIONS")
    print("="*80)
    
    # Load data
    print("\n[1/5] Loading data...")
    movies = load_movies()
    ratings = load_ratings()
    watchlists = load_watchlists()
    
    print(f"   Movies: {len(movies)}")
    print(f"   Ratings: {len(ratings)}")
    print(f"   Watchlists: {len(watchlists)}")
    
    # Initialize models
    print("\n[2/5] Initializing models...")
    factory = ModelFactory(
        movies=movies,
        ratings=ratings,
        watchlists=watchlists,
        normalization='zscore',
        item_k=20,
        user_k=30
    )
    
    models = factory.create_all_models(C=1.0)
    print("   ✅ Models initialized")
    
    # Analyze watchlist overlap
    print("\n[3/5] Analyzing watchlist overlap...")
    overlap_analysis = analyze_watchlist_overlap(watchlists, group_users)
    
    print(f"\n   Individual Watchlist Sizes:")
    for uid, size in overlap_analysis['individual_sizes'].items():
        print(f"      User {uid}: {size} items")
    
    print(f"\n   Common Items: {len(overlap_analysis['common_items'])}")
    print(f"   Total Unique Items: {overlap_analysis['total_unique_items']}")
    print(f"   Overlap Ratio: {overlap_analysis['overlap_ratio']:.2%}")
    
    # Get watched movies (from ratings)
    print("\n[4/5] Identifying watched movies...")
    cf_matrix = models['cf_matrix']
    watched = set()
    
    for uid in group_users:
        if uid in cf_matrix.index:
            user_watched = cf_matrix.loc[uid].dropna().index.tolist()
            watched.update(user_watched)
    
    print(f"   Total watched by group: {len(watched)} movies")
    
    # Get candidates (popular movies not watched)
    all_movies = ratings['movieId'].value_counts().head(500).index.tolist()
    candidates = [m for m in all_movies if m not in watched][:200]
    
    print(f"   Candidate movies: {len(candidates)}")
    
    # Generate recommendations
    print(f"\n[5/5] Generating recommendations for each hybrid model...")
    
    all_recommendations = {}
    
    for model_name in ['h1', 'h2', 'h3']:
        print(f"\n   Generating recommendations with {model_name.upper()}...")
        
        try:
            recs = models[model_name].recommend_for_group(
                group_users, 
                candidates, 
                top_k=top_k
            )
            
            # Enrich with movie details
            enriched_recs = []
            for rec in recs:
                movie_details = get_movie_details(rec['movie_id'], movies)
                enriched_recs.append({
                    **rec,
                    **movie_details
                })
            
            all_recommendations[model_name] = enriched_recs
            print(f"      ✅ Generated {len(enriched_recs)} recommendations")
            
        except Exception as e:
            print(f"      ❌ Failed: {e}")
            all_recommendations[model_name] = []
    
    # Create report
    report = {
        'group_info': {
            'user_ids': group_users,
            'num_users': len(group_users)
        },
        'watchlist_analysis': overlap_analysis,
        'common_watchlist_movies': [
            get_movie_details(mid, movies) 
            for mid in overlap_analysis['common_items'][:10]  # Top 10 common
        ],
        'recommendations': all_recommendations,
        'evaluation_config': {
            'data_source': 'Real users with watchlist data',
            'num_users': len(group_users),
            'total_watchlist_entries': len(watchlists),
            'avg_watchlist_size': len(watchlists) / watchlists['userId'].nunique(),
            'evaluation_type': 'Qualitative case study',
            'top_k': top_k
        }
    }
    
    # Save report
    output_file = os.path.join(
        os.path.dirname(__file__),
        "src", "experiments", "results",
        "case_study_real_users.json"
    )
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Report saved to: {output_file}")
    
    # Print formatted report for paper
    print_paper_formatted_report(report, movies)
    
    return report


def print_paper_formatted_report(report: Dict, movies_df: pd.DataFrame):
    """Print report in a format suitable for the paper."""
    
    print("\n" + "="*80)
    print("FORMATTED REPORT FOR PAPER")
    print("="*80)
    
    # Section 1: Group Information
    print("\n### Case Study: Real User Group")
    print(f"\n**Group Composition:** {report['group_info']['num_users']} users")
    print(f"**User IDs:** {', '.join(map(str, report['group_info']['user_ids']))}")
    
    # Section 2: Watchlist Analysis
    print("\n### Watchlist Analysis")
    print(f"\n**Individual Watchlist Sizes:**")
    for uid, size in report['watchlist_analysis']['individual_sizes'].items():
        print(f"- User {uid}: {size} items")
    
    print(f"\n**Common Watchlist Items:** {len(report['watchlist_analysis']['common_items'])} movies")
    print(f"**Overlap Ratio:** {report['watchlist_analysis']['overlap_ratio']:.1%}")
    
    if report['common_watchlist_movies']:
        print(f"\n**Sample Common Watchlist Movies:**")
        for i, movie in enumerate(report['common_watchlist_movies'][:5], 1):
            print(f"{i}. {movie['title']} ({movie['genres']})")
    
    # Section 3: Recommendations by Model
    print("\n### Group Recommendations")
    
    for model_name in ['h1', 'h2', 'h3']:
        recs = report['recommendations'].get(model_name, [])
        
        if not recs:
            continue
        
        print(f"\n#### Model {model_name.upper()} - Top-{len(recs)} Recommendations")
        print(f"\n| Rank | Movie | Genres | Score | Explanation |")
        print(f"|------|-------|--------|-------|-------------|")
        
        for i, rec in enumerate(recs[:10], 1):
            title = rec['title'][:40] + '...' if len(rec['title']) > 40 else rec['title']
            genres = rec['genres'][:30] + '...' if len(rec['genres']) > 30 else rec['genres']
            score = f"{rec['score']:.3f}"
            
            # Get explanation (truncated)
            explanation = rec.get('explanation', 'N/A')
            if isinstance(explanation, str):
                explanation = explanation[:60] + '...' if len(explanation) > 60 else explanation
            
            print(f"| {i} | {title} | {genres} | {score} | {explanation} |")
    
    # Section 4: Evaluation Configuration
    print("\n### Evaluation Configuration")
    config = report['evaluation_config']
    print(f"\n```")
    print(f"Data Source: {config['data_source']}")
    print(f"Number of Users: {config['num_users']}")
    print(f"Total Watchlist Entries: {config['total_watchlist_entries']}")
    print(f"Average Watchlist Size: {config['avg_watchlist_size']:.1f}")
    print(f"Evaluation Type: {config['evaluation_type']}")
    print(f"Top-K: {config['top_k']}")
    print(f"```")
    
    print("\n" + "="*80)


def main():
    """Run case study for real users."""
    
    # Load watchlists to identify real users
    watchlists = load_watchlists()
    real_users = sorted(watchlists['userId'].unique().tolist())
    
    print(f"\n📊 Identified {len(real_users)} real users: {real_users}")
    
    if len(real_users) < 2:
        print("\n⚠️  Not enough real users for group recommendation.")
        print("   Need at least 2 users.")
        return
    
    # Generate case study for all real users as a group
    print(f"\n🎯 Generating case study for group: {real_users}")
    
    report = generate_case_study_report(
        group_users=real_users,
        top_k=10
    )
    
    print("\n✅ Case study complete!")
    print("\n💡 Use this report in your paper's evaluation section.")
    print("   - Shows real-world applicability")
    print("   - Demonstrates watchlist integration")
    print("   - Provides qualitative analysis")


if __name__ == "__main__":
    main()
