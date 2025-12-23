"""
Film Filter Agent using SmolAgents
Analyzes user prompts and filters candidate movies based on sentiment and explicit criteria.
"""

import os
import json
from typing import List, Dict, Any, Optional
import pandas as pd
from smolagents import CodeAgent, tool, LiteLLMModel, InferenceClientModel, TransformersModel

# Import config manager for keys
try:
    from src.utils.config_manager import get_api_key
except ImportError:
    # Fallback if running standalone or path issues
    def get_api_key(service):
        return os.getenv(f"{service.upper()}_API_KEY") or os.getenv(f"{service.upper()}_TOKEN")

# Define tools

@tool
def analyze_user_preferences(prompt: str) -> Dict[str, Any]:
    """
    Analyzes user's prompt to extract filtering preferences.
    Only extracts criteria that are explicitly mentioned.
    
    Args:
        prompt: User's description of what they want to watch
        
    Returns:
        Dictionary with optional filters:
        - genres: List of genres if mentioned (e.g. ['Action', 'Comedy'])
        - mood: Mood/tone if mentioned (dark, light, intense, calm, psychological)
        - themes: Themes if mentioned
        - actors: Specific actors if mentioned
        - director: Specific director if mentioned
        - min_runtime: Minimum runtime if mentioned
        - max_runtime: Maximum runtime if mentioned
    """
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
        mood: Filter by mood/keywords (if specified) - valid values: 'dark', 'light', 'intense', 'calm', 'psychological'
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
    
    # Pre-process mood keywords
    mood_keywords = {
        'dark': ['dark', 'noir', 'grim', 'sinister', 'bleak', 'horror', 'thriller'],
        'light': ['light', 'cheerful', 'uplifting', 'feel-good', 'heartwarming', 'comedy'],
        'intense': ['intense', 'thriller', 'suspense', 'action', 'fast-paced'],
        'calm': ['calm', 'peaceful', 'gentle', 'slow-paced', 'drama'],
        'psychological': ['psychological', 'mind', 'mental', 'psycho', 'mystery']
    }
    
    for movie in movies:
        passes = True
        
        # Safe get for fields
        m_genres = [g.lower() for g in movie.get('genres', [])]
        m_keywords = str(movie.get('keywords', '')).lower()
        m_overview = str(movie.get('overview', '')).lower()
        m_actors = str(movie.get('actors', '')).lower()
        m_director = str(movie.get('director', '')).lower()
        
        # Genre filter
        if genres and passes:
            # Check if ANY of the requested genres are present
            if not any(g.lower() in m_genres for g in genres):
                passes = False
        
        # Mood filter
        if mood and passes:
            mood_lower = mood.lower()
            if mood_lower in mood_keywords:
                # Check keywords and overview for mood words
                target_words = mood_keywords[mood_lower]
                if not any(w in m_keywords or w in m_overview for w in target_words):
                    passes = False
        
        # Themes filter
        if themes and passes:
            if not any(t.lower() in m_keywords or t.lower() in m_overview for t in themes):
                passes = False
        
        # Actor filter
        if actors and passes:
            if actors.lower() not in m_actors:
                passes = False
        
        # Director filter
        if director and passes:
            if director.lower() not in m_director:
                passes = False
        
        # Runtime filters
        m_runtime = movie.get('runtime')
        if min_runtime and passes:
            if m_runtime is None or m_runtime < min_runtime:
                passes = False
        
        if max_runtime and passes:
            if m_runtime is None or m_runtime > max_runtime:
                passes = False
        
        if passes:
            filtered_ids.append(movie.get('movieId'))
    
    return filtered_ids


class FilmFilterAgent:
    """
    AI Agent that filters movies based on user's natural language preferences
    using smolagents.
    """
    
    def __init__(self):
        """Initialize the Film Filter Agent"""
        self.model = self._setup_model()
        
        # Create agent with tools
        self.agent = CodeAgent(
            tools=[filter_movies], 
            model=self.model,
            max_steps=5
            # add_base_tools=True removed to avoid unneeded dependencies like ddgs
        )
    
    def _setup_model(self):
        """Configure the LLM model"""
        gemini_key = get_api_key('gemini')
        hf_token = get_api_key('huggingface')
        
        # Option 1: Gemini via LiteLLM
        if gemini_key:
            try:
                import litellm
                print("[FilmFilterAgent] Using Gemini model via LiteLLM")
                return LiteLLMModel(
                    model_id="gemini/gemini-1.5-flash",
                    api_key=gemini_key
                )
            except ImportError:
                print("[FilmFilterAgent] Warning: Gemini key found but 'litellm' not installed.")
                print("To use Gemini, run: pip install litellm")

        # Option 2: HuggingFace (Inference API)
        # Use InferenceClientModel instead of TransformersModel to avoid local download!
        if hf_token:
            print("[FilmFilterAgent] Using HuggingFace Inference API")
            try:
                # Use Qwen2.5-Coder-32B via API (fast, no download)
                return InferenceClientModel(
                    model_id="Qwen/Qwen2.5-Coder-32B-Instruct", 
                    token=hf_token
                )
            except Exception as e:
                print(f"[FilmFilterAgent] Failed to initialize InferenceClientModel: {e}")
                print("Falling back to smaller model...")

        # Fallback
        print("[FilmFilterAgent] Warning: Using fallback model via Inference API")
        return InferenceClientModel(
            model_id="Qwen/Qwen2.5-Coder-32B-Instruct",  # Try same model, might differ on public API limits
            token=hf_token if hf_token else None
        )

    def filter_from_prompt(
        self, 
        user_prompt: str, 
        candidate_movies: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Filter movies based on user's natural language prompt.
        
        Args:
            user_prompt: User's description of what they want
            candidate_movies: List of movie dicts with full metadata
            
        Returns:
            Dictionary with filtering results and agent recommendation
        """
        
        task = f"""
You are an expert film curator.
User Request: "{user_prompt}"

I have provided a list of movies in the variable `candidate_movies`. 
Each movie is a dictionary with keys: 'movieId', 'title', 'genres', 'overview', 'keywords', 'actors', 'director', 'runtime', 'vote_average'.

Your goal is to:
1. EXAMINE the user request to identify constraints like genre, mood, actors, director, or runtime.
   - PRINT your thought process using print() to explain what you are looking for.
2. CALL the `filter_movies` tool with the appropriate arguments based on the user request. 
   - Use 'mood' argument if the user asks for 'dark', 'light', 'intense', 'calm', or 'psychological' movies.
   - Use 'genres' list if they mention genres.
   - Use 'actors' or 'director' strings if specific names are invalid.
3. OBTAIN the list of filtered movie IDs.
   - PRINT how many movies remained after filtering.
4. SELECT the single best movie recommendation from the filtered list (or from the originals if filtering was too strict).
   - Use `candidate_movies` to look up details for the selected ID.
   - PRINT the title of the movie you selected.
   - Write a short 'reason' string explaining why it fits.
5. PREPARE a final dictionary with exactly this structure:
   {{
       "filtered_movie_ids": <list of ints>,
       "filters_applied": <dict of args used in filter_movies>,
       "agent_recommendation": {{
            "movieId": <int>,
            "title": <str>,
            "reason": <str>
       }}
   }}
6. RETURN this dictionary as the final result.
"""
        
        try:
            # We must pass the candidate_movies to the agent's context
            result = self.agent.run(task, additional_args={'candidate_movies': candidate_movies})
            
            # Validate result structure
            if isinstance(result, dict) and "filtered_movie_ids" in result:
                # Add stats
                result["total_candidates"] = len(candidate_movies)
                result["filtered_count"] = len(result["filtered_movie_ids"])
                return result
            else:
                print(f"[FilmFilterAgent] Agent returned unexpected format: {result}")
                # Try to salvage if it's just missing stats
                if isinstance(result, dict):
                    result["total_candidates"] = len(candidate_movies)
                    result["filtered_count"] = len(result.get("filtered_movie_ids", []))
                    return result
                    
                raise ValueError("Agent did not return a valid dictionary")
                
        except Exception as e:
            print(f"[FilmFilterAgent] Error: {e}")
            # Fallback logic
            return {
                "filtered_movie_ids": [m['movieId'] for m in candidate_movies],
                "filters_applied": {"error": str(e)},
                "agent_recommendation": None,
                "total_candidates": len(candidate_movies),
                "filtered_count": len(candidate_movies)
            }

# Singleton accessor
_agent_instance = None

def get_filter_agent() -> FilmFilterAgent:
    """Get singleton instance"""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = FilmFilterAgent()
    return _agent_instance
