"""
Film Curator Agent using CrewAI
Analyzes user preferences and filters/recommends movies
"""

from crewai import Agent, Task, Crew
from typing import List, Dict, Any
import os
import json
import re


class FilmCuratorCrewAI:
    """Film curator using CrewAI"""
    
    def __init__(self):
        """Initialize the film curator agent - GEMINI ONLY"""
        # Use centralized config manager
        import sys
        from pathlib import Path
        
        # Add utils to path
        utils_path = Path(__file__).parent.parent / 'utils'
        if str(utils_path) not in sys.path:
            sys.path.insert(0, str(utils_path))
        
        from config_manager import get_api_key
        
        # Get ONLY Gemini API key
        gemini_key = get_api_key('gemini')
        
        if not gemini_key:
            raise ValueError(
                "Gemini API key not found!\n"
                "Please set GEMINI_API_KEY or GOOGLE_API_KEY in your .env file"
            )
        
        # IMPORTANT: Remove any OpenAI keys from environment to force Gemini usage
        if 'OPENAI_API_KEY' in os.environ:
            del os.environ['OPENAI_API_KEY']
        if 'OPENAI_API_BASE' in os.environ:
            del os.environ['OPENAI_API_BASE']
        
        # Set ONLY Gemini key
        os.environ['GOOGLE_API_KEY'] = gemini_key
        
        # Configure CrewAI to use Gemini explicitly
        from crewai import LLM
        
        self.llm = LLM(
            model="gemini/gemini-1.5-flash",
            api_key=gemini_key
        )
        
        self.llm_provider = "Gemini"
        print(f"[CrewAI] ✓ Using Gemini (gemini-1.5-flash) as LLM provider")
        
        # Create the film curator agent with explicit LLM
        self.curator = Agent(
            role='Film Curator',
            goal='Analyze user preferences and recommend the perfect movie from candidates',
            backstory="""You are an expert film curator with deep knowledge of cinema. 
            You understand genres, themes, moods, and can match user preferences to films perfectly.
            You analyze user requests carefully and provide thoughtful recommendations.
            You must always respond in English, regardless of the language used in the user's request.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm  # Explicitly use Gemini LLM
        )
        
        print(f"[CrewAI] ✓ Film Curator Agent initialized successfully with {self.llm_provider}")
    
    def filter_from_prompt(
        self, 
        user_prompt: str, 
        candidate_movies: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Filter and recommend movies based on user prompt
        
        Args:
            user_prompt: User's preference description
            candidate_movies: List of candidate movies to filter
            
        Returns:
            Dict with filtered_movie_ids, filters_applied, and agent_recommendation
        """
        try:
            # Format movies for the agent
            movies_text = "\n\n".join([
                f"Movie {i+1}:\n"
                f"  ID: {m['movieId']}\n"
                f"  Title: {m['title']}\n"
                f"  Genres: {', '.join(m.get('genres', []))}\n"
                f"  Director: {m.get('director', 'Unknown')}\n"
                f"  Overview: {m.get('overview', 'N/A')}\n"
                f"  Keywords: {m.get('keywords', 'N/A')}\n"
                f"  Rating: {m.get('vote_average', 'N/A')}/10"
                for i, m in enumerate(candidate_movies)
            ])
            
            # Create the task
            task = Task(
                description=f"""
                IMPORTANT: You must respond ONLY in English, regardless of the language used in the user's request.
                
                User Request: "{user_prompt}"
                
                Available Movies:
                {movies_text}
                
                Your task:
                1. Analyze the user's request to understand their preferences
                2. Filter the movies that match these preferences
                3. Pick the BEST movie from the filtered list
                4. Explain WHY it's the best choice
                
                Return your response in this EXACT JSON format:
                {{
                    "filtered_movie_ids": [list of matching movie IDs],
                    "filters_applied": {{
                        "genres": [list of genres user wants],
                        "mood": "description of mood",
                        "themes": [list of themes]
                    }},
                    "agent_recommendation": {{
                        "movieId": ID of your top pick,
                        "title": "movie title",
                        "reason": "detailed explanation why this is the best choice"
                    }}
                }}
                """,
                agent=self.curator,
                expected_output="JSON object with filtered movies and recommendation"
            )
            
            # Create crew and execute
            crew = Crew(
                agents=[self.curator],
                tasks=[task],
                verbose=True
            )
            
            result = crew.kickoff()
            
            # Parse the result
            result_text = str(result)
            
            print(f"[CrewAI] Raw agent output length: {len(result_text)} chars")
            
            # Try multiple JSON extraction strategies
            parsed_result = None
            
            # Strategy 1: Direct JSON parsing
            try:
                parsed_result = json.loads(result_text)
                print("[CrewAI] Successfully parsed JSON directly")
            except json.JSONDecodeError:
                pass
            
            # Strategy 2: Find JSON block with curly braces
            if parsed_result is None:
                try:
                    start = result_text.find('{')
                    end = result_text.rfind('}') + 1
                    if start != -1 and end > start:
                        json_str = result_text[start:end]
                        parsed_result = json.loads(json_str)
                        print("[CrewAI] Successfully extracted JSON from text")
                except (json.JSONDecodeError, ValueError):
                    pass
            
            # Strategy 3: Use regex to find JSON block
            if parsed_result is None:
                try:
                    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
                    matches = re.findall(json_pattern, result_text, re.DOTALL)
                    for match in matches:
                        try:
                            parsed_result = json.loads(match)
                            if 'filtered_movie_ids' in parsed_result or 'agent_recommendation' in parsed_result:
                                print("[CrewAI] Successfully extracted JSON using regex")
                                break
                        except json.JSONDecodeError:
                            continue
                except Exception:
                    pass
            
            # If we got valid JSON, add metadata and return
            if parsed_result and isinstance(parsed_result, dict):
                parsed_result['total_candidates'] = len(candidate_movies)
                parsed_result['filtered_count'] = len(parsed_result.get('filtered_movie_ids', []))
                return parsed_result
            
            # Fallback: Return structured error with first 3 movies
            print(f"[CrewAI] Failed to parse JSON from agent response")
            print(f"[CrewAI] Raw response preview: {result_text[:500]}...")
            
            return {
                "error": "Failed to parse agent response",
                "raw_response": result_text[:1000],  # Limit to first 1000 chars
                "filtered_movie_ids": [m['movieId'] for m in candidate_movies[:3]],
                "filters_applied": {"raw_prompt": user_prompt},
                "agent_recommendation": {
                    "movieId": candidate_movies[0]['movieId'],
                    "title": candidate_movies[0]['title'],
                    "reason": "Fallback recommendation (agent response parsing failed)"
                },
                "total_candidates": len(candidate_movies),
                "filtered_count": min(3, len(candidate_movies))
            }
                
        except Exception as e:
            print(f"Agent error: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
            
            return {
                "error": f"{type(e).__name__}: {str(e)}",
                "filtered_movie_ids": [],
                "filters_applied": {},
                "agent_recommendation": None,
                "total_candidates": len(candidate_movies),
                "filtered_count": 0
            }


# Singleton instance
_agent_instance = None

def get_filter_agent():
    """Get or create the film curator agent"""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = FilmCuratorCrewAI()
    return _agent_instance
