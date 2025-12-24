"""
Group Recommendation API with AI Agent Integration
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from src.utils.model_utils import quick_setup
from src.pipeline.structured_output_generator import StructuredOutputGenerator

router = APIRouter(
    prefix="/recommend",
    tags=["Recommendations"]
)


class GroupRecommendationRequest(BaseModel):
    user_ids: List[int]
    trust_mode: bool = True
    custom_prompt: Optional[str] = None
    top_k: int = 10


# NEW: Updated to match full 3-section output
class GroupRecommendationResponse(BaseModel):
    section_a_top_recommendations: List[Dict[str, Any]]
    section_b_common_watchlist: List[Dict[str, Any]]
    section_c_shared_interests: List[Dict[str, Any]]


# Global models cache to avoid reloading
_models_cache = None

def get_models():
    global _models_cache
    if _models_cache is None:
        print("[API] Initializing Recommendation Models...")
        _models_cache = quick_setup(
            recent_only=False,  # FIXED: Include all ratings for power users
            recent_count=50000,
            normalization='zscore',
            item_k=20,
            user_k=20,  
            C=1.0
        )
    return _models_cache


@router.post("/group")
async def generate_group_recommendation(request: GroupRecommendationRequest):
    """
    Generate AI-curated group recommendations
    """
    try:
        print(f"[API] Generating recommendations for users: {request.user_ids}")
        
        # 1. Get Models
        models = get_models()
        
        # 2. Get AI Agent (if needed)
        filter_agent = None
        user_prompt = None
        
        if not request.trust_mode and request.custom_prompt:
            print(f"[API] Custom prompt received: {request.custom_prompt}")
            try:
                from src.agents.film_curator import get_filter_agent
                filter_agent = get_filter_agent()
                user_prompt = request.custom_prompt
            except Exception as e:
                print(f"[API] Warning: Could not init agent: {e}")
        
        # 3. Initialize Structured Output Generator
        generator = StructuredOutputGenerator(
            hybrid_model_1=models['h1'],
            hybrid_model_2=models['h2'],
            hybrid_model_3=models['h3'],
            movies_df=models['movies'],
            watchlist_df=models['watchlists'],
            cf_matrix=models['cf_matrix'],
            ratings_df=models['ratings'],
            enable_temporal_filtering=True,
            film_agent=filter_agent  # Pass agent instance
        )
        
        # 4. Generate Output (3 Sections)
        output = generator.generate_three_section_output(
            group_users=request.user_ids,
            user_prompt=user_prompt
        )
        
        # Fix NaN values for JSON serialization
        import pandas as pd
        import numpy as np
        
        def clean_nan(obj):
            """Recursively replace NaN with None for JSON compatibility"""
            if isinstance(obj, dict):
                return {k: clean_nan(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_nan(item) for item in obj]
            elif isinstance(obj, (float, np.floating)):
                if pd.isna(obj) or np.isinf(obj):
                    return None
                return obj
            elif pd.isna(obj):  # Catch pandas NA types
                return None
            return obj
        
        output = clean_nan(output)
        
        # Return exact dictionary matched to Pydantic model implicitly
        return output

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
    return {"status": "Backend is working!"}


@router.get("/status")
async def get_status():
    """Check if recommendation service is ready"""
    return {
        "status": "Recommendation Service Ready",
        "agent": "SmolAgents + Hybrid Engine"
    }
