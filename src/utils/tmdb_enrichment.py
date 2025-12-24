"""
TMDB Enrichment Service
-----------------------
On-demand fetching of movie posters, backdrops, and trailers from TMDB API.
Caches results to movies_tmdb.csv to avoid redundant API calls.

Usage:
    from src.utils.tmdb_enrichment import enrich_movies_batch
    
    movies = [
        {"movie_id": 1, "title": "Toy Story (1995)"},
        {"movie_id": 2, "title": "Jumanji (1995)"}
    ]
    
    enriched = enrich_movies_batch(movies)
    # Returns movies with poster_url, backdrop_url, trailer_url added
"""

import os
import requests
import pandas as pd
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# TMDB API Configuration
API_KEY = "e04b517a6f93b6cc57636573c1230c91"
BASE_URL = "https://api.themoviedb.org/3"
POSTER_BASE_URL = "https://image.tmdb.org/t/p/w500"  # 500px width for posters
BACKDROP_BASE_URL = "https://image.tmdb.org/t/p/original"  # Original size for backdrops

# Cache file
MOVIES_CSV_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'movies_tmdb.csv')


def search_movie_on_tmdb(title: str) -> Optional[int]:
    """
    Search for a movie on TMDB by title and return its TMDB ID.
    
    Args:
        title: Movie title (e.g., "Toy Story (1995)")
        
    Returns:
        TMDB movie ID or None if not found
    """
    # Clean title (remove year if present)
    clean_title = title.split('(')[0].strip()
    
    try:
        response = requests.get(
            f'{BASE_URL}/search/movie',
            params={'api_key': API_KEY, 'query': clean_title},
            timeout=5
        )
        response.raise_for_status()
        results = response.json().get('results', [])
        
        if results:
            return results[0]['id']  # Return first match
        return None
    except Exception as e:
        print(f"⚠️ TMDB search failed for '{title}': {e}")
        return None


def fetch_movie_data(tmdb_id: int) -> Dict[str, Optional[str]]:
    """
    Fetch poster, backdrop, and trailer from TMDB API.
    
    Args:
        tmdb_id: TMDB movie ID
        
    Returns:
        Dict with poster_path, backdrop_path, trailer_key
    """
    data = {
        'poster_path': None,
        'backdrop_path': None,
        'trailer_key': None
    }
    
    try:
        # Get movie details (poster + backdrop)
        details_response = requests.get(
            f'{BASE_URL}/movie/{tmdb_id}',
            params={'api_key': API_KEY},
            timeout=5
        )
        details_response.raise_for_status()
        details = details_response.json()
        
        data['poster_path'] = details.get('poster_path', '')
        data['backdrop_path'] = details.get('backdrop_path', '')
        
        # Get videos (trailers)
        videos_response = requests.get(
            f'{BASE_URL}/movie/{tmdb_id}/videos',
            params={'api_key': API_KEY},
            timeout=5
        )
        videos_response.raise_for_status()
        videos = videos_response.json().get('results', [])
        
        # Find first YouTube trailer
        trailers = [v for v in videos if v['type'] == 'Trailer' and v['site'] == 'YouTube']
        if trailers:
            data['trailer_key'] = trailers[0]['key']
        
    except Exception as e:
        print(f"⚠️ TMDB fetch failed for ID {tmdb_id}: {e}")
    
    return data


def update_csv_cache(movie_id: int, tmdb_data: Dict[str, Optional[str]]):
    """
    Update the movies_tmdb.csv file with fetched TMDB data.
    
    Args:
        movie_id: MovieLens movie ID
        tmdb_data: Dict with poster_path, backdrop_path, trailer_key
    """
    try:
        df = pd.read_csv(MOVIES_CSV_PATH)
        
        # Find the movie row
        mask = df['movieId'] == movie_id
        if mask.any():
            # Convert values to string to avoid dtype issues
            poster = str(tmdb_data.get('poster_path', '')) if tmdb_data.get('poster_path') else ''
            backdrop = str(tmdb_data.get('backdrop_path', '')) if tmdb_data.get('backdrop_path') else ''
            trailer = str(tmdb_data.get('trailer_key', '')) if tmdb_data.get('trailer_key') else ''
            
            # Update the row
            df.loc[mask, 'poster_path'] = poster
            df.loc[mask, 'backdrop_path'] = backdrop
            df.loc[mask, 'trailer_key'] = trailer
            
            # Save back to CSV
            df.to_csv(MOVIES_CSV_PATH, index=False)
            print(f"✅ Cached TMDB data for movie {movie_id}")
    except Exception as e:
        print(f"⚠️ Failed to update CSV cache for movie {movie_id}: {e}")


def get_cached_data(movie_id: int) -> Optional[Dict[str, str]]:
    """
    Check if TMDB data is already cached in CSV.
    
    Args:
        movie_id: MovieLens movie ID
        
    Returns:
        Dict with poster_path, backdrop_path, trailer_key or None if not cached
    """
    try:
        df = pd.read_csv(MOVIES_CSV_PATH)
        row = df[df['movieId'] == movie_id]
        
        if not row.empty:
            poster = row.iloc[0].get('poster_path', '')
            backdrop = row.iloc[0].get('backdrop_path', '')
            trailer = row.iloc[0].get('trailer_key', '')
            
            # Convert NaN/None to empty string
            poster = '' if pd.isna(poster) else str(poster)
            backdrop = '' if pd.isna(backdrop) else str(backdrop)
            trailer = '' if pd.isna(trailer) else str(trailer)
            
            # Check if ANY data exists (not all empty)
            # Only return cached data if at least one field has a value
            if poster or backdrop or trailer:
                return {
                    'poster_path': poster,
                    'backdrop_path': backdrop,
                    'trailer_key': trailer
                }
        return None
    except Exception as e:
        print(f"⚠️ Failed to read CSV cache: {e}")
        return None


def enrich_movie(movie_id: int, title: str) -> tuple[Dict[str, Optional[str]], Dict[str, Optional[str]]]:
    """
    Enrich a single movie with TMDB data (cache-first).
    
    Args:
        movie_id: MovieLens movie ID
        title: Movie title
        
    Returns:
        Tuple of (url_dict, raw_tmdb_data):
            - url_dict: Dict with poster_url, backdrop_url, trailer_url
            - raw_tmdb_data: Dict with poster_path, backdrop_path, trailer_key (for caching)
    """
    # Check cache first
    cached = get_cached_data(movie_id)
    if cached:
        print(f"💾 Cache hit for movie {movie_id}: {title}")
        return (
            {
                'poster_url': f"{POSTER_BASE_URL}{cached['poster_path']}" if cached['poster_path'] else None,
                'backdrop_url': f"{BACKDROP_BASE_URL}{cached['backdrop_path']}" if cached['backdrop_path'] else None,
                'trailer_url': f"https://www.youtube.com/watch?v={cached['trailer_key']}" if cached['trailer_key'] else None
            },
            cached  # Return cached raw data
        )
    
    # Cache miss - fetch from TMDB
    print(f"🔍 Fetching TMDB data for movie {movie_id}: {title}")
    tmdb_id = search_movie_on_tmdb(title)
    
    if not tmdb_id:
        empty_data = {'poster_path': None, 'backdrop_path': None, 'trailer_key': None}
        return (
            {'poster_url': None, 'backdrop_url': None, 'trailer_url': None},
            empty_data
        )
    
    tmdb_data = fetch_movie_data(tmdb_id)
    
    # Return URLs and raw data (caller will handle caching)
    return (
        {
            'poster_url': f"{POSTER_BASE_URL}{tmdb_data['poster_path']}" if tmdb_data['poster_path'] else None,
            'backdrop_url': f"{BACKDROP_BASE_URL}{tmdb_data['backdrop_path']}" if tmdb_data['backdrop_path'] else None,
            'trailer_url': f"https://www.youtube.com/watch?v={tmdb_data['trailer_key']}" if tmdb_data['trailer_key'] else None
        },
        tmdb_data
    )


def enrich_movies_batch(movies: List[Dict], max_workers: int = 5) -> List[Dict]:
    """
    Enrich multiple movies in parallel.
    
    Args:
        movies: List of dicts with 'movie_id' and 'title' keys
        max_workers: Number of parallel API calls (default: 5)
        
    Returns:
        List of movies with poster_url, backdrop_url, trailer_url added
    """
    print(f"\n🎬 Enriching {len(movies)} movies with TMDB data...")
    
    enriched_movies = []
    updates_to_cache = {}  # Collect updates: {movie_id: tmdb_data}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_movie = {
            executor.submit(enrich_movie, m['movie_id'], m['title']): m
            for m in movies
        }
        
        # Collect results
        for future in as_completed(future_to_movie):
            movie = future_to_movie[future]
            try:
                url_data, raw_tmdb_data = future.result()
                # Add URL data to movie dict
                movie.update(url_data)
                enriched_movies.append(movie)
                
                # Collect raw TMDB data for batch caching (only if not from cache)
                if raw_tmdb_data and any(raw_tmdb_data.values()):
                    updates_to_cache[movie['movie_id']] = raw_tmdb_data
            except Exception as e:
                print(f"⚠️ Failed to enrich movie {movie['movie_id']}: {e}")
                # Add movie without TMDB data
                movie.update({'poster_url': None, 'backdrop_url': None, 'trailer_url': None})
                enriched_movies.append(movie)
    
    # Batch update CSV (only once, for all new data)
    if updates_to_cache:
        print(f"\n💾 Updating CSV cache with {len(updates_to_cache)} new entries...")
        try:
            df = pd.read_csv(MOVIES_CSV_PATH)
            
            for movie_id, tmdb_data in updates_to_cache.items():
                mask = df['movieId'] == movie_id
                if mask.any():
                    poster = str(tmdb_data.get('poster_path', '')) if tmdb_data.get('poster_path') else ''
                    backdrop = str(tmdb_data.get('backdrop_path', '')) if tmdb_data.get('backdrop_path') else ''
                    trailer = str(tmdb_data.get('trailer_key', '')) if tmdb_data.get('trailer_key') else ''
                    
                    df.loc[mask, 'poster_path'] = poster
                    df.loc[mask, 'backdrop_path'] = backdrop
                    df.loc[mask, 'trailer_key'] = trailer
            
            # Write CSV only once
            df.to_csv(MOVIES_CSV_PATH, index=False, quoting=1)  # quoting=1 = QUOTE_MINIMAL
            print(f"✅ CSV cache updated successfully!")
        except Exception as e:
            print(f"⚠️ Failed to update CSV cache: {e}")
    
    print(f"✅ Enrichment complete!\n")
    return enriched_movies


# Test function
if __name__ == "__main__":
    # Test with a few movies
    test_movies = [
        {"movie_id": 1, "title": "Toy Story (1995)"},
        {"movie_id": 2, "title": "Jumanji (1995)"},
        {"movie_id": 3, "title": "Grumpier Old Men (1995)"}
    ]
    
    enriched = enrich_movies_batch(test_movies)
    
    print("\n=== Results ===")
    for movie in enriched:
        print(f"\n{movie['title']}:")
        print(f"  Poster: {movie['poster_url']}")
        print(f"  Backdrop: {movie['backdrop_url']}")
        print(f"  Trailer: {movie['trailer_url']}")
