"""
Group Recommendation API with AI Agent Integration
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional

router = APIRouter(
    prefix="/recommend",
    tags=["Recommendations"]
)


class GroupRecommendationRequest(BaseModel):
    user_ids: List[int]
    trust_mode: bool = True
    custom_prompt: Optional[str] = None
    top_k: int = 10


class MovieRecommendation(BaseModel):
    movieId: int
    title: str
    reason: str
    score: Optional[float] = None
    genres: Optional[List[str]] = None
    director: Optional[str] = None
    actors: Optional[str] = None
    overview: Optional[str] = None
    runtime: Optional[int] = None
    vote_average: Optional[float] = None
    keywords: Optional[str] = None
    production_countries: Optional[str] = None


class GroupRecommendationResponse(BaseModel):
    agent_recommendation: MovieRecommendation  # Agent's top pick (required)
    filters_applied: Optional[dict] = None  # What filters were used
    total_candidates: Optional[int] = None  # How many movies were considered
    filtered_count: Optional[int] = None  # How many passed filters
    method: str  # "trust" or "custom_prompt"


@router.post("/group")
async def generate_group_recommendation(request: GroupRecommendationRequest):
    """
    Generate AI-curated group recommendations
    
    Flow:
    1. Get base recommendations from Hybrid Model
    2. If trust_mode: Return top hybrid recommendations
    3. If custom_prompt: Agent filters based on sentiment + provides its own pick
    """
    
    try:
        # AI Agent test with mock movies
        print(f"[API] Testing AI Agent for users: {request.user_ids}")
        
        # Mock movie candidates (simulating hybrid model output)
        mock_movies = [
            {
                "movieId": 1,
                "title": "The Shawshank Redemption",
                "genres": ["Crime", "Drama"],
                "director": "Frank Darabont",
                "actors": "Tim Robbins, Morgan Freeman",
                "overview": "Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency.",
                "runtime": 142,
                "vote_average": 8.7,
                "keywords": "prison, friendship, hope, redemption",
                "production_countries": "United States"
            },
            {
                "movieId": 2,
                "title": "The Dark Knight",
                "genres": ["Action", "Crime", "Drama", "Thriller"],
                "director": "Christopher Nolan",
                "actors": "Christian Bale, Heath Ledger, Aaron Eckhart",
                "overview": "When the menace known as the Joker wreaks havoc and chaos on the people of Gotham, Batman must accept one of the greatest psychological and physical tests.",
                "runtime": 152,
                "vote_average": 9.0,
                "keywords": "dc comics, crime fighter, terrorist, secret identity, crime, superhero, psychological",
                "production_countries": "United States"
            },
            {
                "movieId": 3,
                "title": "Inception",
                "genres": ["Action", "Science Fiction", "Thriller"],
                "director": "Christopher Nolan",
                "actors": "Leonardo DiCaprio, Joseph Gordon-Levitt, Ellen Page",
                "overview": "Cobb, a skilled thief who commits corporate espionage by infiltrating the subconscious of his targets is offered a chance to regain his old life.",
                "runtime": 148,
                "vote_average": 8.8,
                "keywords": "dream, subconscious, mission, heist, psychological",
                "production_countries": "United States"
            },
            {
                "movieId": 4,
                "title": "Pulp Fiction",
                "genres": ["Thriller", "Crime"],
                "director": "Quentin Tarantino",
                "actors": "John Travolta, Uma Thurman, Samuel L. Jackson",
                "overview": "A burger-loving hit man, his philosophical partner, a drug-addled gangster's moll and a washed-up boxer converge in this sprawling, comedic crime caper.",
                "runtime": 154,
                "vote_average": 8.9,
                "keywords": "drug dealer, boxer, massage, los angeles, violence, dark",
                "production_countries": "United States"
            },
            {
                "movieId": 5,
                "title": "Forrest Gump",
                "genres": ["Comedy", "Drama", "Romance"],
                "director": "Robert Zemeckis",
                "actors": "Tom Hanks, Robin Wright, Gary Sinise",
                "overview": "A man with a low IQ has accomplished great things in his life and been present during significant historic events.",
                "runtime": 142,
                "vote_average": 8.8,
                "keywords": "vietnam veteran, hippie, mentally disabled, friendship, usa, washington dc",
                "production_countries": "United States"
            }
        ]
        
        if request.trust_mode:
            # Trust mode: Return first mock movie
            recommendation = MovieRecommendation(
                movieId=mock_movies[0]['movieId'],
                title=mock_movies[0]['title'],
                reason="Top pick from mock data (trust mode)",
                score=0.95,
                genres=mock_movies[0]['genres'],
                director=mock_movies[0]['director'],
                actors=mock_movies[0]['actors'],
                overview=mock_movies[0]['overview'],
                runtime=mock_movies[0]['runtime'],
                vote_average=mock_movies[0]['vote_average'],
                keywords=mock_movies[0]['keywords'],
                production_countries=mock_movies[0]['production_countries']
            )
            
            return GroupRecommendationResponse(
                agent_recommendation=recommendation,
                filters_applied=None,
                total_candidates=len(mock_movies),
                filtered_count=len(mock_movies),
                method="trust"
            )
        else:
            # Custom prompt mode: Use AI Agent
            if not request.custom_prompt:
                raise HTTPException(
                    status_code=400,
                    detail="Custom prompt required when trust_mode is False"
                )
            
            print(f"[API] ========================================")
            print(f"[API] AI Agent analyzing prompt: {request.custom_prompt}")
            print(f"[API] Number of mock movies: {len(mock_movies)}")
            print(f"[API] ========================================")
            
            try:
                # Get AI agent
                import sys
                import os
                sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                
                print("[API] Importing CrewAI filter agent...")
                from agents.film_curator_crewai import get_filter_agent
                
                print("[API] Getting filter agent instance...")
                filter_agent = get_filter_agent()
                
                print("[API] Calling agent.filter_from_prompt()...")
                # Agent filters and recommends
                filter_result = filter_agent.filter_from_prompt(
                    user_prompt=request.custom_prompt,
                    candidate_movies=mock_movies
                )
                
                print(f"[API] ===== AGENT RAW OUTPUT =====")
                print(f"Type: {type(filter_result)}")
                print(f"Content: {filter_result}")
                print(f"[API] ==============================")
                
                # Return agent's raw output as-is
                return filter_result
                
            except Exception as e:
                print(f"[API] !!!!! AGENT ERROR !!!!!")
                print(f"[API] Error type: {type(e).__name__}")
                print(f"[API] Error message: {str(e)}")
                import traceback
                print(f"[API] Full traceback:")
                traceback.print_exc()
                print(f"[API] !!!!!!!!!!!!!!!!!!!!!!!!!")
                
                raise HTTPException(
                    status_code=500,
                    detail=f"Agent error: {type(e).__name__}: {str(e)}"
                )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[API ERROR] Recommendation failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate recommendations: {str(e)}"
        )


@router.get("/test")
async def test_endpoint():
    """Simple test endpoint"""
    print("[API] ===== TEST ENDPOINT CALLED =====")
    return {"status": "Backend is working!", "message": "If you see this, the API is running correctly"}


@router.get("/status")
async def get_status():
    """Check if recommendation service is ready"""
    return {
        "status": "Recommendation Service Ready",
        "agent": "SmolAgents Film Filter",
        "model": "meta-llama/Llama-3.3-70B-Instruct"
    }
