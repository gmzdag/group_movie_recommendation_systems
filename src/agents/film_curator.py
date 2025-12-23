"""
Film Filter Agent using SmolAgents
Analyzes user prompts and filters candidate movies based on sentiment
"""

from smolagents import CodeAgent, tool
from smolagents.models import TransformersModel
from typing import List, Dict, Any, Optional
import pandas as pd
import os

# Define custom tools for the agent

@tool
def analyze_user_preferences(prompt: str) -> Dict[str, Any]:
    """
    Analyzes user's prompt to extract filtering preferences.
    Only extracts criteria that are explicitly mentioned.
    
    Args:
        prompt: User's description of what they want to watch
        
    Returns:
        Dictionary with optional filters:
        - genres: List of genres if mentioned
        - mood: Mood/tone if mentioned (dark, light, intense, etc.)
        - themes: Themes if mentioned
        - actors: Specific actors if mentioned
        - director: Specific director if mentioned
        - min_runtime: Minimum runtime if mentioned
        - max_runtime: Maximum runtime if mentioned
    """
    # This will be analyzed by the LLM
    # Only return filters that are explicitly in the prompt
    return {
        "genres": None,
        "mood": None,
        "themes": None,
        "actors": None,
        "director": None,
        "min_runtime": None,
        "max_runtime": None
    }


@tool
def filter_movies(
    movies: List[Dict[str, Any]],
    genres: Optional[List[str]] = None,
    mood: Optional[str] = None,
    themes: Optional[List[str]] = None,
    actors: Optional[str] = None,
    director: Optional[str] = None,
    min_runtime: Optional[int] = None,
    max_runtime: Optional[int] = None
) -> List[int]:
    """
    Filters movies based on provided criteria.
    Only applies filters that are not None.
    
    Args:
        movies: List of movie dictionaries with full metadata
        genres: Filter by genres (if specified)
        mood: Filter by mood/keywords (if specified)
        themes: Filter by themes/keywords (if specified)
        actors: Filter by actor name (if specified)
        director: Filter by director name (if specified)
        min_runtime: Minimum runtime in minutes (if specified)
        max_runtime: Maximum runtime in minutes (if specified)
        
    Returns:
        List of movieIds that pass all specified filters
    """
    if not movies:
        return []
    
    filtered_ids = []
    
    for movie in movies:
        # Start with True, will become False if any filter fails
        passes = True
        
        # Genre filter (if specified)
        if genres and passes:
            movie_genres = [g.lower() for g in movie.get('genres', [])]
            if not any(g.lower() in movie_genres for g in genres):
                passes = False
        
        # Mood filter via keywords (if specified)
        if mood and passes:
            keywords = movie.get('keywords', '').lower()
            overview = movie.get('overview', '').lower()
            
            # Map moods to keyword patterns
            mood_keywords = {
                'dark': ['dark', 'noir', 'grim', 'sinister', 'bleak'],
                'light': ['light', 'cheerful', 'uplifting', 'feel-good', 'heartwarming'],
                'intense': ['intense', 'thriller', 'suspense', 'action'],
                'calm': ['calm', 'peaceful', 'gentle', 'slow-paced'],
                'psychological': ['psychological', 'mind', 'mental', 'psycho']
            }
            
            mood_lower = mood.lower()
            if mood_lower in mood_keywords:
                if not any(kw in keywords or kw in overview for kw in mood_keywords[mood_lower]):
                    passes = False
        
        # Themes filter via keywords (if specified)
        if themes and passes:
            keywords = movie.get('keywords', '').lower()
            overview = movie.get('overview', '').lower()
            if not any(theme.lower() in keywords or theme.lower() in overview for theme in themes):
                passes = False
        
        # Actor filter (if specified)
        if actors and passes:
            movie_actors = movie.get('actors', '').lower()
            if actors.lower() not in movie_actors:
                passes = False
        
        # Director filter (if specified)
        if director and passes:
            movie_director = movie.get('director', '').lower()
            if director.lower() not in movie_director:
                passes = False
        
        # Runtime filters (if specified)
        if min_runtime and passes:
            runtime = movie.get('runtime')
            if runtime is None or runtime < min_runtime:
                passes = False
        
        if max_runtime and passes:
            runtime = movie.get('runtime')
            if runtime is None or runtime > max_runtime:
                passes = False
        
        # If passed all filters, add to results
        if passes:
            filtered_ids.append(movie['movieId'])
    
    return filtered_ids


# Utility function for loading movie data
def get_available_movies(movie_ids: List[int]) -> List[Dict[str, Any]]:
    """
    Retrieves full movie details for given movie IDs from movies_tmdb.csv.
    
    Args:
        movie_ids: List of movie IDs to retrieve
        
    Returns:
        List of movie dictionaries with all available metadata
    """
    # Load movies data
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    movies_file = os.path.join(data_dir, "movies_tmdb.csv")
    
    if not os.path.exists(movies_file):
        return []
    
    movies_df = pd.read_csv(movies_file)
    
    # Filter by movie IDs
    filtered = movies_df[movies_df['movieId'].isin(movie_ids)]
    
    # Convert to list of dicts with all columns
    movies = []
    for _, row in filtered.iterrows():
        movie = {
            "movieId": int(row['movieId']),
            "title": row['title'],
            "genres": row.get('genres', '').split('|') if pd.notna(row.get('genres')) else [],
            "director": row.get('Director', 'Unknown'),
            "actors": row.get('Actors', 'Unknown'),
            "overview": row.get('Overview', ''),
            "runtime": int(row['Runtime']) if pd.notna(row.get('Runtime')) else None,
            "vote_average": float(row['Vote_Average']) if pd.notna(row.get('Vote_Average')) else None,
            "keywords": row.get('Keywords', ''),
            "production_countries": row.get('Production_Countries', '')
        }
        movies.append(movie)
    
    return movies


class FilmFilterAgent:
    """
    AI Agent that filters movies based on user's natural language preferences
    Does NOT recommend - only analyzes sentiment and filters
    """
    
    def __init__(self, model_id: str = "meta-llama/Llama-3.3-70B-Instruct"):
        """
        Initialize the Film Filter Agent
        
        Args:
            model_id: HuggingFace model ID to use
        """
        # Load HuggingFace token from environment
        from dotenv import load_dotenv
        load_dotenv()
        
        hf_token = os.getenv('HUGGINGFACE_TOKEN')
        if not hf_token or hf_token == 'your_token_here':
            raise ValueError(
                "HuggingFace token not found! "
                "Please set HUGGINGFACE_TOKEN in .env file"
            )
        
        # Use a smaller model for testing
        self.model = TransformersModel(
            model_id="HuggingFaceTB/SmolLM2-1.7B-Instruct",
            token=hf_token
        )
        
        # Create agent with filtering tools
        self.agent = CodeAgent(
            tools=[analyze_user_preferences, filter_movies],
            model=self.model,
            max_steps=3
        )
    
    def filter_from_prompt(
        self, 
        user_prompt: str, 
        candidate_movies: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Filter movies based on user's natural language prompt
        AND provide agent's own recommendation
        
        Args:
            user_prompt: User's description of what they want
            candidate_movies: List of movie dicts with full metadata
            
        Returns:
            Dictionary with:
            - filtered_movie_ids: List of movieIds that match criteria
            - filters_applied: Dict of filters that were applied
            - agent_recommendation: Agent's top pick from filtered results
            - total_candidates: Original number of candidates
            - filtered_count: Number after filtering
        """
        
        task = f"""
IMPORTANT: You must respond ONLY in English, regardless of the language used in the user's request.

You are a film filter and recommendation assistant.

User's request: "{user_prompt}"

Available movies: {len(candidate_movies)} movies with full metadata

Your task:
1. Use analyze_user_preferences to extract ONLY the criteria explicitly mentioned in the prompt
   - If user doesn't mention actors, don't filter by actors
   - If user doesn't mention director, don't filter by director
   - If user doesn't mention runtime, don't filter by runtime
   - Only extract what is CLEARLY stated in the prompt

2. Use filter_movies to apply ONLY the extracted filters to the candidate movies
   - Pass None for any filter not mentioned in the prompt
   - Return the list of movieIds that pass the filters

3. From the filtered results, pick your TOP recommendation that best matches the user's request
   - Consider the movie's overview, keywords, ratings
   - Provide a brief reason why this movie is perfect for their request

Return a JSON with:
{{
    "filtered_movie_ids": [list of all movieIds that passed filters],
    "filters_applied": {{
        "genres": [list] or null,
        "mood": str or null,
        "themes": [list] or null,
        "actors": str or null,
        "director": str or null,
        "min_runtime": int or null,
        "max_runtime": int or null
    }},
    "agent_recommendation": {{
        "movieId": int,
        "title": str,
        "reason": str (why this is the best match)
    }}
}}
"""
        
        try:
            result = self.agent.run(task, candidate_movies=candidate_movies)
            
            # Ensure result has required fields
            if not isinstance(result, dict):
                result = {
                    "filtered_movie_ids": [],
                    "filters_applied": {},
                    "agent_recommendation": None
                }
            
            return {
                "filtered_movie_ids": result.get("filtered_movie_ids", []),
                "filters_applied": result.get("filters_applied", {}),
                "agent_recommendation": result.get("agent_recommendation"),
                "total_candidates": len(candidate_movies),
                "filtered_count": len(result.get("filtered_movie_ids", []))
            }
            
        except Exception as e:
            print(f"[ERROR] Agent failed: {e}")
            # Return all candidates if filtering fails
            return {
                "filtered_movie_ids": [m['movieId'] for m in candidate_movies],
                "filters_applied": {},
                "agent_recommendation": None,
                "total_candidates": len(candidate_movies),
                "filtered_count": len(candidate_movies),
                "error": str(e)
            }


# Singleton instance
_filter_agent = None

def get_filter_agent() -> FilmFilterAgent:
    """Get or create the filter agent singleton"""
    global _filter_agent
    if _filter_agent is None:
        _filter_agent = FilmFilterAgent()
    return _filter_agent
