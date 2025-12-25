"""
META-ENSEMBLE GROUP RECOMMENDER - COMPLETE DEMO
================================================

SYSTEM DEFINITION:
-----------------
Combines 3 independent hybrid models based on offline performance (NDCG@10):
- Hybrid Model 1 (Dynamic Weighted: IBCF + CBF) - BEST
- Hybrid Model 2 (Switching: UBCF → CBF)
- Hybrid Model 3 (Watchlist-Driven)

KEY PRINCIPLES:
1. Models do NOT merge - each operates independently
2. Each movie belongs to exactly ONE source model
3. Selection based on offline performance from data/cache/model_performance.json
4. Better-performing models get priority

STRICT CONSTRAINTS:
------------------
❌ NO watchlist items in Top-10 (watchlist only for analysis)
❌ NO generalizations ("Everyone likes...", "Group loves genre X")
❌ NO score merging between models

OUTPUT STRUCTURE:
----------------
🅰️ PART A: Top-10 Main Group Recommendations
🅲️ PART C: Thematic Group Recommendations (SEPARATE)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from collections import defaultdict, Counter

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.model_utils import quick_setup


def load_model_performance():
    """Load model performance metrics from cache."""
    perf_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "cache", "model_performance.json"
    )
    
    if os.path.exists(perf_path):
        with open(perf_path, 'r') as f:
            perf_data = json.load(f)
        
        print("✅ Loaded model performance from cache:")
        print(f"   H1 (Hybrid 1): NDCG@10 = {perf_data['h1']['ndcg@10']:.4f} [Rank {perf_data['h1']['rank']}]")
        print(f"   H2 (Hybrid 2): NDCG@10 = {perf_data['h2']['ndcg@10']:.4f} [Rank {perf_data['h2']['rank']}]")
        print(f"   H3 (Hybrid 3): NDCG@10 = {perf_data['h3']['ndcg@10']:.4f} [Rank {perf_data['h3']['rank']}]")
        
        return {
            'Hybrid Model 1': perf_data['h1']['ndcg@10'],
            'Hybrid Model 2': perf_data['h2']['ndcg@10'],
            'Hybrid Model 3': perf_data['h3']['ndcg@10']
        }
    else:
        print("⚠️  Model performance file not found, using defaults")
        return {
            'Hybrid Model 1': 0.4523,
            'Hybrid Model 2': 0.3876,
            'Hybrid Model 3': 0.2591
        }


def get_group_watchlist(watchlist_df, user_ids):
    """Get all movies in group members' watchlists."""
    group_wl = watchlist_df[watchlist_df['userId'].isin(user_ids)]['movieId'].unique()
    return set(group_wl)


def get_group_watched(ratings_df, user_ids):
    """Get all movies watched by group members."""
    group_watched = ratings_df[ratings_df['userId'].isin(user_ids)]['movieId'].unique()
    return set(group_watched)


def generate_candidates_from_models(models, user_ids, candidate_pool, model_performances):
    """
    Generate candidates from all 3 models.
    
    Returns:
        List of dicts with movie_id, source_model, model_score, model_performance
    """
    candidates = []
    
    # Model 1: Hybrid 1
    print("\n[Generating] Hybrid Model 1 candidates...")
    try:
        h1_recs = models['h1'].recommend_for_group(
            user_ids, 
            candidates=candidate_pool,
            top_k=30
        )
        for rec in h1_recs:
            candidates.append({
                'movie_id': rec['movie_id'],
                'source_model': 'Hybrid Model 1',
                'model_score': rec['score'],
                'model_performance': model_performances['Hybrid Model 1'],
                'raw_data': rec
            })
        print(f"   ✓ Generated {len(h1_recs)} candidates")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
    
    # Model 2: Hybrid 2
    print("[Generating] Hybrid Model 2 candidates...")
    try:
        h2_recs = models['h2'].recommend_for_group(
            user_ids,
            candidates=candidate_pool,
            top_k=30
        )
        for rec in h2_recs:
            candidates.append({
                'movie_id': rec['movie_id'],
                'source_model': 'Hybrid Model 2',
                'model_score': rec['score'],
                'model_performance': model_performances['Hybrid Model 2'],
                'raw_data': rec
            })
        print(f"   ✓ Generated {len(h2_recs)} candidates")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
    
    # Model 3: Hybrid 3
    print("[Generating] Hybrid Model 3 candidates...")
    try:
        h3_recs = models['h3'].recommend_for_group(
            user_ids,
            candidates=candidate_pool,
            top_k=30
        )
        for rec in h3_recs:
            candidates.append({
                'movie_id': rec['movie_id'],
                'source_model': 'Hybrid Model 3',
                'model_score': rec['score'],
                'model_performance': model_performances['Hybrid Model 3'],
                'raw_data': rec
            })
        print(f"   ✓ Generated {len(h3_recs)} candidates")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
    
    return candidates


def select_top_k_by_performance(candidates, group_watchlist, movies_df, top_k=10):
    """
    Select Top-K movies prioritizing better-performing models.
    
    Strategy:
    1. Filter out watchlist items (STRICT RULE)
    2. Deduplicate: Keep only best model per movie
    3. Sort by: model_performance DESC, then model_score DESC
    4. Take top_k
    """
    movie_id_to_title = movies_df.set_index('movieId')['title'].to_dict()
    movie_id_to_genres = movies_df.set_index('movieId')['genres'].to_dict()
    
    # Deduplicate and filter
    movie_to_best_candidate = {}
    
    for cand in candidates:
        mid = cand['movie_id']
        
        # ❌ STRICT: Skip watchlist items
        if mid in group_watchlist:
            continue
        
        if mid not in movie_to_best_candidate:
            movie_to_best_candidate[mid] = cand
        else:
            existing = movie_to_best_candidate[mid]
            # Prioritize better model performance
            if cand['model_performance'] > existing['model_performance']:
                movie_to_best_candidate[mid] = cand
            elif cand['model_performance'] == existing['model_performance']:
                # Same performance, compare scores
                if cand['model_score'] > existing['model_score']:
                    movie_to_best_candidate[mid] = cand
    
    # Sort by performance, then score
    sorted_candidates = sorted(
        movie_to_best_candidate.values(),
        key=lambda x: (x['model_performance'], x['model_score']),
        reverse=True
    )
    
    # Take top_k
    top_candidates = sorted_candidates[:top_k]
    
    # Format output
    results = []
    for rank, cand in enumerate(top_candidates, 1):
        mid = cand['movie_id']
        title = movie_id_to_title.get(mid, f"Movie {mid}")
        genres = movie_id_to_genres.get(mid, 'Unknown')
        
        results.append({
            'rank': rank,
            'movie': title,
            'movie_id': mid,
            'genres': genres,
            'source_model': cand['source_model'],
            'model_score': round(cand['model_score'], 3),
            'model_performance_ndcg': round(cand['model_performance'], 4)
        })
    
    return results


def generate_explanations(top_10, user_ids, ratings_df, watchlist_df, movies_df):
    """
    Generate multi-dimensional, NON-GENERALIZING explanations.
    
    RULES:
    - NO "Everyone likes..."
    - NO "Group loves genre X"
    - Use specific user behaviors
    - Reference watchlist for context (but not as direct match)
    """
    movie_id_to_genres = movies_df.set_index('movieId')['genres'].to_dict()
    
    for rec in top_10:
        mid = rec['movie_id']
        source_model = rec['source_model']
        movie_genres = set(movie_id_to_genres.get(mid, '').split('|'))
        
        # Why this model recommended this movie
        why_model = []
        if source_model == 'Hybrid Model 1':
            why_model.append("• Dynamic weighted combination of item-based CF and content-based signals")
            why_model.append("• Balances collaborative patterns with content similarity")
        elif source_model == 'Hybrid Model 2':
            why_model.append("• Switching strategy: User-based CF with content-based fallback")
            why_model.append("• Adapts to data availability per user")
        elif source_model == 'Hybrid Model 3':
            why_model.append("• Watchlist-driven content filtering based on future intent")
            why_model.append("• Focuses on what users plan to watch")
        
        # Why suitable for group (NO GENERALIZATIONS)
        why_group = []
        
        # Analyze per-user signals
        for uid in user_ids:
            user_ratings = ratings_df[ratings_df['userId'] == uid]
            
            # Find similar rated movies
            similar_movies = []
            for _, row in user_ratings.iterrows():
                rated_mid = row['movieId']
                rated_genres = set(movie_id_to_genres.get(rated_mid, '').split('|'))
                
                if len(movie_genres & rated_genres) > 0:
                    similar_movies.append({
                        'rating': row['rating'],
                        'overlap': len(movie_genres & rated_genres)
                    })
            
            if similar_movies:
                avg_rating = np.mean([m['rating'] for m in similar_movies])
                if avg_rating >= 3.5:
                    why_group.append(
                        f"• User {uid}: Rated {len(similar_movies)} similar movies "
                        f"with avg {avg_rating:.1f}/5.0"
                    )
        
        # Watchlist context (NOT direct match, since we filtered those out)
        watchlist_similar = 0
        for uid in user_ids:
            user_wl = watchlist_df[watchlist_df['userId'] == uid]['movieId'].values
            for wl_mid in user_wl:
                wl_genres = set(movie_id_to_genres.get(wl_mid, '').split('|'))
                if len(movie_genres & wl_genres) >= 2:
                    watchlist_similar += 1
                    break
        
        if watchlist_similar > 0:
            why_group.append(
                f"• Content overlaps with watchlist items from {watchlist_similar} member(s)"
            )
        
        rec['why_this_model'] = why_model
        rec['why_suitable_for_group'] = why_group if why_group else ["• Recommended based on model's internal scoring"]
    
    return top_10


def generate_thematic_recommendations(user_ids, ratings_df, watchlist_df, movies_df, 
                                     group_watched, group_watchlist):
    """
    Generate thematic recommendations (Part C).
    
    Themes:
    1. Genre-Based
    2. Year-Based (Decade)
    """
    themes = []
    
    # 1. Genre-Based Theme
    genre_theme = generate_genre_theme(user_ids, ratings_df, movies_df, group_watched, group_watchlist)
    if genre_theme:
        themes.append(genre_theme)
    
    # 2. Year-Based Theme
    year_theme = generate_year_theme(user_ids, ratings_df, movies_df, group_watched, group_watchlist)
    if year_theme:
        themes.append(year_theme)
    
    return themes


def generate_genre_theme(user_ids, ratings_df, movies_df, group_watched, group_watchlist):
    """Generate genre-based thematic recommendation."""
    movie_id_to_genres = movies_df.set_index('movieId')['genres'].to_dict()
    
    # Analyze genre preferences per user
    user_genre_stats = {}
    
    for uid in user_ids:
        user_ratings = ratings_df[ratings_df['userId'] == uid]
        user_avg = user_ratings['rating'].mean()
        
        genre_ratings = defaultdict(list)
        for _, row in user_ratings.iterrows():
            mid = row['movieId']
            rating = row['rating']
            genres = movie_id_to_genres.get(mid, '').split('|')
            
            for genre in genres:
                if genre and genre != '(no genres listed)':
                    genre_ratings[genre].append(rating)
        
        # Find genres rated above user average
        above_avg_genres = {}
        for genre, ratings in genre_ratings.items():
            if len(ratings) >= 3:
                avg_genre_rating = np.mean(ratings)
                if avg_genre_rating > user_avg:
                    above_avg_genres[genre] = {
                        'avg_rating': avg_genre_rating,
                        'count': len(ratings)
                    }
        
        user_genre_stats[uid] = above_avg_genres
    
    # Find common genres (at least 2 users)
    genre_counter = Counter()
    for uid, genres in user_genre_stats.items():
        for genre in genres.keys():
            genre_counter[genre] += 1
    
    common_genres = [g for g, count in genre_counter.items() if count >= 2]
    
    if not common_genres:
        return None
    
    # Pick top genre
    selected_genre = common_genres[0]
    
    # Find unwatched movies in this genre
    genre_movies = []
    for _, row in movies_df.iterrows():
        mid = row['movieId']
        if mid in group_watched or mid in group_watchlist:
            continue
        
        genres = row.get('genres', '').split('|')
        if selected_genre in genres:
            genre_movies.append({
                'movie_id': mid,
                'title': row['title']
            })
    
    if not genre_movies:
        return None
    
    recommendations = genre_movies[:5]
    
    # Generate explanation
    supporting_users = [
        uid for uid, genres in user_genre_stats.items() 
        if selected_genre in genres
    ]
    
    why_selected = [
        f"User {uid}: Rated {user_genre_stats[uid][selected_genre]['count']} "
        f"{selected_genre} movies with avg {user_genre_stats[uid][selected_genre]['avg_rating']:.1f}/5.0 "
        f"(above personal average)"
        for uid in supporting_users
    ]
    
    return {
        'theme': f'{selected_genre} Movies',
        'why_selected': why_selected,
        'recommendations': recommendations
    }


def generate_year_theme(user_ids, ratings_df, movies_df, group_watched, group_watchlist):
    """Generate year-based (decade) thematic recommendation."""
    # Extract years from titles
    movie_to_year = {}
    for _, row in movies_df.iterrows():
        mid = row['movieId']
        title = row['title']
        if '(' in title and ')' in title:
            year_str = title[title.rfind('(')+1:title.rfind(')')]
            try:
                movie_to_year[mid] = int(year_str)
            except:
                pass
    
    # Analyze decade preferences
    user_decade_stats = {}
    
    for uid in user_ids:
        user_ratings = ratings_df[ratings_df['userId'] == uid]
        user_avg = user_ratings['rating'].mean()
        
        decade_ratings = defaultdict(list)
        for _, row in user_ratings.iterrows():
            mid = row['movieId']
            rating = row['rating']
            year = movie_to_year.get(mid)
            
            if year:
                decade = (year // 10) * 10
                decade_ratings[decade].append(rating)
        
        # Find decades rated above average
        above_avg_decades = {}
        for decade, ratings in decade_ratings.items():
            if len(ratings) >= 3:
                avg_decade_rating = np.mean(ratings)
                if avg_decade_rating > user_avg:
                    above_avg_decades[decade] = {
                        'avg_rating': avg_decade_rating,
                        'count': len(ratings)
                    }
        
        user_decade_stats[uid] = above_avg_decades
    
    # Find common decades
    decade_counter = Counter()
    for uid, decades in user_decade_stats.items():
        for decade in decades.keys():
            decade_counter[decade] += 1
    
    common_decades = [d for d, count in decade_counter.items() if count >= 2]
    
    if not common_decades:
        return None
    
    selected_decade = common_decades[0]
    
    # Find unwatched movies from this decade
    decade_movies = []
    for mid, year in movie_to_year.items():
        if mid in group_watched or mid in group_watchlist:
            continue
        
        if (year // 10) * 10 == selected_decade:
            title = movies_df[movies_df['movieId'] == mid]['title'].values
            if len(title) > 0:
                decade_movies.append({
                    'movie_id': mid,
                    'title': title[0]
                })
    
    if not decade_movies:
        return None
    
    recommendations = decade_movies[:5]
    
    supporting_users = [
        uid for uid, decades in user_decade_stats.items()
        if selected_decade in decades
    ]
    
    why_selected = [
        f"User {uid}: Rated {user_decade_stats[uid][selected_decade]['count']} "
        f"{selected_decade}s movies with avg {user_decade_stats[uid][selected_decade]['avg_rating']:.1f}/5.0"
        for uid in supporting_users
    ]
    
    return {
        'theme': f'{selected_decade}s Movies',
        'why_selected': why_selected,
        'recommendations': recommendations
    }


def format_output(top_10, thematic):
    """Format output according to specification."""
    print("\n" + "=" * 80)
    print("🅰️ PART A — TOP-10 MAIN GROUP RECOMMENDATIONS")
    print("=" * 80)
    
    for rec in top_10:
        print(f"\nRank: {rec['rank']}")
        print(f"Movie: {rec['movie']}")
        print(f"Source Model: {rec['source_model']}")
        print(f"Model Score: {rec['model_score']}")
        print(f"Model Performance (NDCG@10): {rec['model_performance_ndcg']}")
        
        print("\nWhy This Model Recommended This Movie:")
        for reason in rec['why_this_model']:
            print(f"  {reason}")
        
        print("\nWhy This Movie Is Suitable For The Group:")
        for reason in rec['why_suitable_for_group']:
            print(f"  {reason}")
        print("-" * 80)
    
    print("\n" + "=" * 80)
    print("🅲️ PART C — THEMATIC GROUP RECOMMENDATIONS")
    print("=" * 80)
    
    for i, theme in enumerate(thematic, 1):
        print(f"\n{i}️⃣ Theme: {theme['theme']}")
        
        print("\nWhy This Theme Was Selected:")
        for reason in theme['why_selected']:
            print(f"  • {reason}")
        
        print("\nRecommendations:")
        for movie in theme['recommendations']:
            print(f"  • {movie['title']}")
        print("-" * 80)


def main():
    """Run Meta-Ensemble Group Recommender Demo."""
    print("=" * 80)
    print("META-ENSEMBLE GROUP RECOMMENDER - COMPLETE DEMO")
    print("=" * 80)
    
    # Load model performance
    print("\n[SETUP] Loading model performance metrics...")
    model_performances = load_model_performance()
    
    # Initialize models
    print("\n[SETUP] Initializing hybrid models...")
    models = quick_setup(
        recent_only=False,
        normalization='zscore',
        item_k=20,
        user_k=30,
        C=1.0
    )
    
    # Test group
    group_users = [618, 623]
    print(f"\n[GROUP] Testing with users: {group_users}")
    
    # Get watchlist and watched items
    print("\n[FILTERING] Identifying watchlist and watched items...")
    group_watchlist = get_group_watchlist(models['watchlists'], group_users)
    group_watched = get_group_watched(models['ratings'], group_users)
    
    print(f"   • Watchlist items: {len(group_watchlist)}")
    print(f"   • Watched items: {len(group_watched)}")
    
    # Generate candidate pool
    all_movies = set(models['movies']['movieId'].unique())
    candidate_pool = list(all_movies - group_watched - group_watchlist)[:500]
    print(f"   • Candidate pool size: {len(candidate_pool)}")
    
    # Generate candidates from all models
    print("\n[GENERATION] Generating candidates from all 3 models...")
    candidates = generate_candidates_from_models(
        models, 
        group_users, 
        candidate_pool,
        model_performances
    )
    print(f"\n   Total candidates generated: {len(candidates)}")
    
    # Select Top-10 by performance
    print("\n[SELECTION] Selecting Top-10 by model performance...")
    top_10 = select_top_k_by_performance(
        candidates,
        group_watchlist,
        models['movies'],
        top_k=10
    )
    print(f"   ✓ Selected {len(top_10)} recommendations")
    
    # Generate explanations
    print("\n[EXPLANATION] Generating multi-dimensional explanations...")
    top_10 = generate_explanations(
        top_10,
        group_users,
        models['ratings'],
        models['watchlists'],
        models['movies']
    )
    
    # Generate thematic recommendations
    print("\n[THEMATIC] Generating thematic recommendations...")
    thematic = generate_thematic_recommendations(
        group_users,
        models['ratings'],
        models['watchlists'],
        models['movies'],
        group_watched,
        group_watchlist
    )
    print(f"   ✓ Generated {len(thematic)} themes")
    
    # Format and display output
    format_output(top_10, thematic)
    
    # Save output
    output_file = os.path.join(
        os.path.dirname(__file__), 
        "..", 
        "meta_ensemble_output.json"
    )
    
    output_data = {
        'part_a_top_10': top_10,
        'part_c_thematic': thematic,
        'metadata': {
            'group_users': group_users,
            'model_performances': model_performances,
            'watchlist_count': len(group_watchlist),
            'watched_count': len(group_watched)
        }
    }
    
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_to_json_serializable(obj):
        """Recursively convert numpy types to native Python types."""
        if isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    output_data_serializable = convert_to_json_serializable(output_data)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data_serializable, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 80)
    print(f"✅ Output saved to: {output_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
