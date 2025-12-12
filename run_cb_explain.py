"""
RUN SCRIPT – CONTENT-BASED EXPLANABILITY TEST (CSV OUTPUT)
----------------------------------------------------------
Selects 10 target movies and finds their top-2 most similar movies.
Saves the results to 'cb_similarity_analysis.csv'.
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src.recommender.data_loader import load_movies, load_ratings
from src.recommender.content_based import ContentBasedModel

def explain_similarity(movie_a, movie_b):
    """
    Compares two movie metadata rows and returns the commonality.
    """
    reasons = []
    
    # 1. Genres
    genres_a = set(str(movie_a['genres']).split('|'))
    genres_b = set(str(movie_b['genres']).split('|'))
    common_genres = genres_a.intersection(genres_b)
    if common_genres:
        reasons.append(f"Genres[{len(common_genres)}]")
        
    # 2. Director
    dir_a = str(movie_a.get('Director', '')).replace(" ", "").lower()
    dir_b = str(movie_b.get('Director', '')).replace(" ", "").lower()
    if dir_a == dir_b and dir_a not in ['nan', '']:
        reasons.append(f"Director({movie_a['Director']})")
        
    # 3. Actors (Overlap)
    actors_a = [x.strip().replace(" ", "").lower() for x in str(movie_a.get('Actors', '')).split(',')]
    actors_b = [x.strip().replace(" ", "").lower() for x in str(movie_b.get('Actors', '')).split(',')]
    common_actors = set(actors_a).intersection(set(actors_b))
    if common_actors and 'nan' not in common_actors:
        reasons.append(f"Actors[{len(common_actors)}]")
        
    # 4. Keywords
    kw_a = set(str(movie_a.get('Keywords', '')).split('|'))
    kw_b = set(str(movie_b.get('Keywords', '')).split('|'))
    common_kw = kw_a.intersection(kw_b)
    if common_kw and '' not in common_kw:
        reasons.append(f"Keywords[{len(common_kw)}]")
        
    return " + ".join(reasons)

# ==========================================================
# 1. SETUP
# ==========================================================
print("\n[STEP 1] Loading Data & Model...")
movies = load_movies()
movies_indexed = movies.set_index('movieId')
ratings = pd.DataFrame(columns=['userId', 'movieId', 'rating']) # Dummy
cb = ContentBasedModel(movies, ratings)

# ==========================================================
# 2. SELECT 10 TARGET MOVIES
# ==========================================================
# Trying to pick popular movies by title string match
search_queries = [
    "Toy Story",        # Animation
    "Dark Knight",      # Action/Crime (Christopher Nolan)
    "Godfather",        # Crime/Drama
    "Inception",        # Sci-Fi (Christopher Nolan)
    "Titanic",          # Romance
    "Matrix",           # Sci-Fi
    "Pulp Fiction",     # Crime/Indie
    "Forrest Gump",     # Drama
    "Lion King",        # Animation
    "Fight Club"        # Thriller
]

target_ids = []
print(f"\n[STEP 2] Searching for {len(search_queries)} target movies...")

for q in search_queries:
    # Contains query, usually safe
    match = movies[movies['title'].str.contains(q, case=False, na=False)]
    if not match.empty:
        # Prefer shortest title usually (exact match)
        # Sort by length of title
        match = match.assign(len_title=match['title'].str.len())
        best = match.sort_values('len_title').iloc[0]
        target_ids.append(best['movieId'])
        print(f"  Found: {best['title']} (ID: {best['movieId']})")
    else:
        print(f"  Warning: Could not find '{q}'")

# ==========================================================
# 3. FIND SIMILAR & SAVE CSV
# ==========================================================
results = []

print(f"\n[STEP 3] analyzing similarities...")

for mid in target_ids:
    if mid not in cb.movie_to_idx:
        continue
        
    idx = cb.movie_to_idx[mid]
    target_vec = cb.tfidf_matrix[idx]
    
    # Compute similarity with ALL movies
    sims = cosine_similarity(target_vec, cb.tfidf_matrix).flatten()
    
    # Get top 3 (0 is itself)
    top_indices = sims.argsort()[::-1][1:3]
    
    target_meta = movies_indexed.loc[mid]
    
    for rank, sim_idx in enumerate(top_indices):
        sim_row = cb.movies_df.iloc[sim_idx]
        score = sims[sim_idx]
        
        explanation = explain_similarity(target_meta, sim_row)
        
        results.append({
            "Target Movie": target_meta['title'],
            "Target Genres": target_meta['genres'],
            "Similar Movie": sim_row['title'],
            "Score": round(score, 4),
            "Explanation": explanation,
            "Sim Genres": sim_row['genres']
        })

# Save
df_res = pd.DataFrame(results)
print(df_res)
df_res.to_csv("cb_similarity_analysis.csv", index=False)
print(f"\nSaved results to 'cb_similarity_analysis.csv'")
