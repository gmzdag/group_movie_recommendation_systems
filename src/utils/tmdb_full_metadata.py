"""
TMDB Full Metadata Fetcher
---------------------------
Fetches complete movie metadata including genres, director, cast, etc.
"""

import requests
from typing import Dict, Optional

API_KEY = "8265bd1679663a7ea12ac168da84d2e8"
BASE_URL = "https://api.themoviedb.org/3"

def fetch_full_movie_metadata(tmdb_id: int) -> Dict[str, Optional[str]]:
    """
    Fetch complete movie metadata from TMDB API.
    
    Returns:
        Dict with ALL metadata fields for movies_tmdb.csv
    """
    data = {
        'genres': '',
        'Director': '',
        'Actors': '',
        'Overview': '',
        'Production_Countries': '',
        'Production_Companies': '',
        'Vote_Average': '',
        'Vote_Count': '',
        'Budget': '',
        'Revenue': '',
        'Keywords': '',
        'Runtime': '',
        'Content_Type': 'Movie',  # Default
        'release_date': '',
        'poster_path': '',
        'backdrop_path': '',
        'trailer_key': ''
    }
    
    try:
        # 1. Get movie details
        details_response = requests.get(
            f'{BASE_URL}/movie/{tmdb_id}',
            params={'api_key': API_KEY},
            timeout=10
        )
        details_response.raise_for_status()
        details = details_response.json()
        
        # Extract basic info
        data['poster_path'] = details.get('poster_path', '')
        data['backdrop_path'] = details.get('backdrop_path', '')
        data['Overview'] = details.get('overview', '')
        data['release_date'] = details.get('release_date', '')
        data['Runtime'] = str(details.get('runtime', '')) if details.get('runtime') else ''
        data['Vote_Average'] = str(details.get('vote_average', '')) if details.get('vote_average') else ''
        data['Vote_Count'] = str(details.get('vote_count', '')) if details.get('vote_count') else ''
        data['Budget'] = str(details.get('budget', '')) if details.get('budget') else ''
        data['Revenue'] = str(details.get('revenue', '')) if details.get('revenue') else ''
        
        # Genres
        genres = details.get('genres', [])
        data['genres'] = '|'.join([g['name'] for g in genres]) if genres else ''
        
        # Countries
        countries = details.get('production_countries', [])
        data['Production_Countries'] = '|'.join([c['name'] for c in countries]) if countries else ''
        
        # Production Companies
        companies = details.get('production_companies', [])
        data['Production_Companies'] = '|'.join([c['name'] for c in companies]) if companies else ''
        
        # 2. Get credits (director + cast)
        credits_response = requests.get(
            f'{BASE_URL}/movie/{tmdb_id}/credits',
            params={'api_key': API_KEY},
            timeout=10
        )
        credits_response.raise_for_status()
        credits = credits_response.json()
        
        # Director
        crew = credits.get('crew', [])
        directors = [c['name'] for c in crew if c.get('job') == 'Director']
        data['Director'] = directors[0] if directors else ''
        
        # Cast (top 5)
        cast = credits.get('cast', [])
        top_cast = [c['name'] for c in cast[:5]]
        data['Actors'] = ', '.join(top_cast) if top_cast else ''
        
        # 3. Get keywords
        keywords_response = requests.get(
            f'{BASE_URL}/movie/{tmdb_id}/keywords',
            params={'api_key': API_KEY},
            timeout=10
        )
        keywords_response.raise_for_status()
        keywords_data = keywords_response.json()
        keywords = keywords_data.get('keywords', [])
        data['Keywords'] = '|'.join([k['name'] for k in keywords]) if keywords else ''
        
        # 4. Get videos (trailers)
        videos_response = requests.get(
            f'{BASE_URL}/movie/{tmdb_id}/videos',
            params={'api_key': API_KEY},
            timeout=10
        )
        videos_response.raise_for_status()
        videos = videos_response.json().get('results', [])
        
        # Find first YouTube trailer
        trailers = [v for v in videos if v['type'] == 'Trailer' and v['site'] == 'YouTube']
        if trailers:
            data['trailer_key'] = trailers[0]['key']
        
    except Exception as e:
        print(f"⚠️ TMDB full metadata fetch failed for ID {tmdb_id}: {e}")
    
    return data
