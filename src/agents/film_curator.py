"""
Film Filter Agent - Refactored Architecture
-------------------------------------------
A deterministic, 3-layer architecture for filtering movies based on user intent.
Layers:
1. LLMFilterExtractor: Extracts intent from prompt (strictly JSON, no logic).
2. NormalizationLayer: standardized synonyms, countries, and genres.
3. DeterministicMovieFilter: Evidence-based filtering against metadata and text.

Design guarantees reproducibility and strict adherence to specific constraints (Year, Country, etc.)
while allowing flexible text matching for themes.
"""

import os
import json
import re
import ast
from typing import List, Dict, Any, Optional, Set
import pandas as pd
from smolagents import LiteLLMModel, InferenceClientModel

# Import config manager for keys
try:
    from src.utils.config_manager import get_api_key
except ImportError:
    # Fallback if running standalone or path issues
    def get_api_key(service):
        return os.getenv(f"{service.upper()}_API_KEY") or os.getenv(f"{service.upper()}_TOKEN")

# ============================================================================
# LAYER 1: LLM INTENT EXTRACTOR
# ============================================================================

class LLMFilterExtractor:
    """
    Layer 1: Pure intent extraction.
    Uses LLM to convert natural language into a strict JSON schema.
    Does NOT access movie data. Does NOT hallucinations (prompt-constrained).
    """
    
    def __init__(self, model):
        self.model = model
        
    def extract(self, user_prompt: str) -> Dict[str, Any]:
        """
        Extract structured constraints from user prompt.
        """
        system_prompt = """
You are a precision movie query parser.
Your GOAL: Extract filtering constraints significantly explicitly stated in the user prompt.

INPUT: User search query.
OUTPUT: Strict JSON object.

SCHEMA:
{
  "genres": [string] | null,
  "mood": string | null,
  "themes": [string] | null,
  "actors": string | null,
  "director": string | null,
  "year_min": integer | null,
  "year_max": integer | null,
  "country": string | null,
  "min_runtime": integer | null,
  "max_runtime": integer | null
}

RULES:
1. ONLY extract what is EXPLICITLY requested.
2. DO NOT hallucinate. If attributes are not mentioned, return null.
3. DO NOT infer genres from themes (e.g., "funny" -> mood="funny", NOT genres=["Comedy"]).
4. "French movies" -> country="France". 
5. "90s movies" -> year_min=1990, year_max=1999.
6. "Short movies" -> max_runtime=90 (approx partial rule allowed for relative terms).

EXAMPLES:
"French action movies" -> {"country": "France", "genres": ["Action"]}
"Yılbaşı filmleri" -> {"themes": ["Yılbaşı"]}
"Comedy about weddings" -> {"genres": ["Comedy"], "themes": ["wedding"]}
"""
        
        # Prepare the message for the model
        # Note: smolagents models typically accept a list of messages or a prompt string.
        # We will wrap it in a simple prompt format if the model expects string, or messages if it supports chat.
        # InferenceClientModel usually handles chat templates if passed messages?
        # Let's try passing a constructed prompt string to be safe across generic models.
        
        full_prompt = f"{system_prompt}\n\nUSER PROMPT: {user_prompt}\n\nJSON RESPONSE:"
        
        for attempt in range(2):
            try:
                # Call model directly
                if hasattr(self.model, "generate"):
                    response = self.model.generate(messages=[{"role": "user", "content": full_prompt}])
                    if hasattr(response, "content"):
                        response_text = response.content
                    else:
                        response_text = str(response)
                else:
                    response_text = str(self.model(full_prompt))
                
                parsed = self._parse_json(response_text)
                if parsed is not None:
                    return parsed
                print(f"[LLMFilterExtractor] Parsing failed on attempt {attempt+1}, retrying...")
                
            except Exception as e:
                print(f"[LLMFilterExtractor] Error calling model (attempt {attempt+1}): {e}")
                
        return self._empty_criteria()

    def _parse_json(self, response_text: str) -> Optional[Dict[str, Any]]:
        """Robust JSON parsing. Returns None if parsing fails completely."""
        try:
            # 1. Clean markdown code blocks
            clean_text = re.sub(r'```json\s*', '', response_text)
            clean_text = re.sub(r'```python\s*', '', clean_text)
            clean_text = re.sub(r'```', '', clean_text).strip()
            
            # 2. Extract JSON-like block
            match = re.search(r'\{.*\}', clean_text, re.DOTALL)
            if match:
                json_str = match.group()
            else:
                if '{' in clean_text: 
                    json_str = clean_text 
                else: 
                    return None

            # 3. Try standard JSON parse
            return json.loads(json_str)
            
        except json.JSONDecodeError:
            # 4. Fallback: Python syntax
            try:
                sanitized = json_str.replace('null', 'None') \
                                  .replace('true', 'True') \
                                  .replace('false', 'False')
                return ast.literal_eval(sanitized)
            except Exception:
                return None
    
    def _empty_criteria(self) -> Dict[str, Any]:
        return {
             "genres": None, "mood": None, "themes": None, 
             "actors": None, "director": None, 
             "year_min": None, "year_max": None, 
             "country": None, 
             "min_runtime": None, "max_runtime": None
        }


# ============================================================================
# LAYER 2: NORMALIZATION LAYER
# ============================================================================

class NormalizationLayer:
    """
    Layer 2: Deterministic Normalization using strict Python mappings.
    Handles Multilingual support (TR->EN), Synonyms, and standardization.
    """
    
    def __init__(self):
        self.country_map = {
            'fransa': 'France', 'fransız': 'France', 'french': 'France', 'france': 'France',
            'türkiye': 'Turkey', 'türk': 'Turkey', 'turkish': 'Turkey', 'turkey': 'Turkey',
            'amerika': 'United States', 'abd': 'United States', 'usa': 'United States', 'united states': 'United States',
            'ingiltere': 'United Kingdom', 'uk': 'United Kingdom', 'british': 'United Kingdom',
            'almanya': 'Germany', 'german': 'Germany',
            'güney kore': 'South Korea', 'korean': 'South Korea',
            'hindistan': 'India', 'indian': 'India',
            'japonya': 'Japan', 'japanese': 'Japan'
        }
        
        self.genre_map = {
            'bilim kurgu': 'Science Fiction', 'scifi': 'Science Fiction', 'sci-fi': 'Science Fiction',
            'aksiyon': 'Action',
            'komedi': 'Comedy',
            'dram': 'Drama',
            'korku': 'Horror',
            'romantik': 'Romance', 'aşk': 'Romance', 'ask': 'Romance',
            'macera': 'Adventure',
            'suç': 'Crime', 'polisiye': 'Crime',
            'animasyon': 'Animation',
            'aile': 'Family'
        }
        
        self.theme_map = {
            'yılbaşı': ['Christmas', 'New Year', 'Holiday'],
            'noel': ['Christmas'],
            'christmas': ['Christmas', 'Holiday'],
            'uzay': ['Space', 'Alien'],
            'space': ['Space'],
            'savaş': ['War', 'Military'],
            'war': ['War'],
        }

    def normalize(self, criteria: Dict[str, Any]) -> Dict[str, Any]:
        """Apply normalization rules to the criteria dictionary."""
        normalized = criteria.copy()
        
        # 1. Normalize Country
        if normalized.get('country') and isinstance(normalized['country'], str):
            c_raw = normalized['country'].strip().lower()
            # Check direct map and partial matches
            if c_raw in self.country_map:
                normalized['country'] = self.country_map[c_raw]
            else:
                # Try partial match (e.g. "french films" -> "french")
                for key, val in self.country_map.items():
                    if key in c_raw:
                        normalized['country'] = val
                        break

        # 2. Normalize Genres
        if normalized.get('genres'):
            raw_genres = normalized['genres']
            if isinstance(raw_genres, str): raw_genres = [raw_genres]
            
            norm_genres = set()
            for g in raw_genres:
                g_str = str(g).lower().strip()
                if g_str in self.genre_map:
                    norm_genres.add(self.genre_map[g_str])
                else:
                    norm_genres.add(g) # Keep original if no map
            
            normalized['genres'] = list(norm_genres)

        # 3. Normalize Themes (Expansion)
        if normalized.get('themes'):
            raw_themes = normalized['themes']
            if isinstance(raw_themes, str): raw_themes = [raw_themes]
            
            expanded_themes = set()
            for t in raw_themes:
                t_str = str(t).lower().strip()
                # exact map
                if t_str in self.theme_map:
                    expanded_themes.update(self.theme_map[t_str])
                else:
                    expanded_themes.add(t) # Keep original (e.g. "Vampire")
            
            normalized['themes'] = list(expanded_themes)
            
        return normalized


# ============================================================================
# LAYER 3: DETERMINISTIC MOVIE FILTER (EVIDENCE-BASED)
# ============================================================================

class DeterministicMovieFilter:
    """
    Layer 3: Evidence-Based Filtering.
    Checks metadata columns and text fields (Overview, Keywords, Title).
    """
    
    def apply(self, movies: List[Dict], criteria: Dict[str, Any]) -> List[int]:
        """
        Filter movies based on normalized criteria.
        Returns list of matching movie IDs.
        """
        filtered_ids = []
        
        # Pre-process criteria for speed
        c_country = criteria.get('country')
        c_year_min = criteria.get('year_min')
        c_year_max = criteria.get('year_max')
        c_genres = criteria.get('genres')
        c_themes = criteria.get('themes')
        c_mood = criteria.get('mood')
        c_actors = criteria.get('actors')
        c_director = criteria.get('director')
        
        # Mood Keywords Registry
        mood_keywords = {
            'dark': ['dark', 'noir', 'grim', 'horror', 'thriller', 'bleak'],
            'light': ['light', 'funny', 'comedy', 'feel-good', 'cheerful'],
            'intense': ['intense', 'action', 'thriller', 'suspense', 'fast'],
            'calm': ['calm', 'drama', 'slow', 'quiet', 'peaceful'],
            'festive': ['christmas', 'holiday', 'family', 'joy'],
            'sad': ['sad', 'drama', 'tragic', 'cry']
        }
        
        target_mood_words = []
        if c_mood and c_mood.lower() in mood_keywords:
            target_mood_words = mood_keywords[c_mood.lower()]
        elif c_mood:
            target_mood_words = [c_mood.lower()] 

        for m in movies:
            passes = True
            
            # --- PREPARE DATA ---
            # Use string concatenation for broad text search
            m_title = str(m.get('title', '')).lower()
            m_overview = str(m.get('overview', '')).lower()
            m_keywords = str(m.get('keywords', '')).lower()
            m_text_blob = f"{m_title} {m_overview} {m_keywords}"
            
            m_countries = str(m.get('countries', '')).lower() # from enriched
            m_year = m.get('year', 0)
            m_genres = [str(g).lower() for g in m.get('genres', [])]
            m_actors = str(m.get('actors', '')).lower()
            m_director = str(m.get('director', '')).lower()
            
            # --- APPLY FILTERS ---

            # 1. Year (Strict Metadata)
            if c_year_min is not None and passes:
                if m_year < c_year_min: passes = False
            if c_year_max is not None and passes:
                if m_year > c_year_max: passes = False
                
            # 2. Country (Metadata OR Text Evidence)
            if c_country and passes:
                tgt = c_country.lower()
                # Matches if in structured 'countries' list OR explicitly mentioned in text
                # e.g. "French cinema" in keywords
                evidence_found = (tgt in m_countries) or (tgt in m_text_blob)
                if not evidence_found:
                    passes = False
            
            # 3. Genres (Metadata OR Text Evidence)
            if c_genres and passes:
                # ALL requested genres must be present (AND logic)? 
                # Usually users mean "Action AND Comedy" -> Rush Hour.
                # Let's enforce AND logic for genres list.
                for g in c_genres:
                    tgt = g.lower()
                    # Check structured genres first
                    genre_match = False
                    # Partial match allows 'Sci-Fi' to match 'Science Fiction' if not normalized
                    if any(tgt == mg or tgt in mg for mg in m_genres):
                        genre_match = True
                    # Check text as fallback
                    elif tgt in m_text_blob: 
                        genre_match = True
                    
                    if not genre_match:
                        passes = False
                        break
            
            # 4. Themes (Text Evidence)
            if c_themes and passes:
                # ANY or ALL? Usually "Christmas" -> check match.
                # If multiple themes: "Space" AND "War"? Or "Space War"?
                # Let's treat list as OR if synonyms, but AND if distinct? 
                # Our normalization produces a list of synonyms for ONE intent usually.
                # But LLM might output ["Christmas", "Family"] -> AND logic?
                # Let's use ANY match from the list, because 'themes' often contains synonyms.
                # Wait, normalization: 'yılbaşı' -> ['Christmas', 'New Year'].
                # User wants a movie matching ANY of those synonyms.
                # But if LLM outputs ['Space', 'Comedy'] (distinct concepts)?
                # Simplification: If ANY theme in the list is found, it's a match.
                # (Assuming the list represents "Possible Topics").
                
                # However, for 'Christmas' AND 'Comedy', 'Comedy' is a genre.
                # If list is ['Christmas', 'Holiday'], they are synonyms.
                # If list is ['Vampire', 'Love'], user likely wants Vampire AND Love?
                # Given strict filtering, OR is safer to avoid zero results, but AND is more precise.
                # Compromise: Match if ANY of the themes in the list is found.
                # (Because Layer 2 tends to expand synonyms into this list).
                
                # Actually, check logic:
                # If normalized themes = ['Christmas', 'New Year', 'Holiday'] (from single 'Yılbaşı') -> OR is correct.
                # If user asked "Vampires and Werewolves" -> LLM ['Vampires', 'Werewolves']. 
                # OR is still probably okay (Vampire movie OR Werewolf movie).
                
                if not any(t.lower() in m_text_blob for t in c_themes):
                    passes = False
            
            # 5. Actors (Text Evidence)
            if c_actors and passes:
                if c_actors.lower() not in m_actors and c_actors.lower() not in m_text_blob:
                    passes = False

            # 6. Director (Text Evidence)
            if c_director and passes:
                if c_director.lower() not in m_director and c_director.lower() not in m_text_blob:
                    passes = False

            # 7. Mood (Keyword Match)
            if c_mood and passes:
                if not any(w in m_text_blob for w in target_mood_words):
                    passes = False

            if passes:
                filtered_ids.append(m['movieId'])
        
        return filtered_ids


# ============================================================================
# ORCHESTRATOR: FILM FILTER AGENT
# ============================================================================

class FilmFilterAgent:
    """
    Orchestrates the 3-layer filtering process.
    """
    
    def __init__(self):
        self.model = self._setup_model()
        
        # Initialize Layers
        self.extractor = LLMFilterExtractor(self.model)
        self.normalizer = NormalizationLayer()
        self.filter_engine = DeterministicMovieFilter()
        
    def _setup_model(self):
        """Configure the LLM model (LiteLLM or HF Inference)."""
        gemini_key = get_api_key('gemini')
        hf_token = get_api_key('huggingface')
        
        # Option 1: Gemini via LiteLLM
        if gemini_key:
            try:
                import litellm
                print("[FilmFilterAgent] Using Gemini model via LiteLLM")
                return LiteLLMModel(model_id="gemini/gemini-1.5-flash", api_key=gemini_key)
            except ImportError:
                pass

        # Option 2: HuggingFace
        if hf_token:
            print("[FilmFilterAgent] Using HuggingFace Inference API")
            try:
                return InferenceClientModel(model_id="Qwen/Qwen2.5-Coder-32B-Instruct", token=hf_token)
            except Exception:
                pass
                
        print("[FilmFilterAgent] Warning: Using fallback model")
        return InferenceClientModel(model_id="Qwen/Qwen2.5-Coder-32B-Instruct", token=hf_token)

    def filter_from_prompt(self, user_prompt: str, candidate_movies: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Execute the 3-layer filtering pipeline.
        """
        print(f"[FilmFilterAgent] Processing prompt: '{user_prompt}'")
        
        # Layer 1: Extract Intent
        raw_criteria = self.extractor.extract(user_prompt)
        print(f"[Layer 1] Extracted: {raw_criteria}")
        
        # Layer 2: Normalize
        norm_criteria = self.normalizer.normalize(raw_criteria)
        print(f"[Layer 2] Normalized: {norm_criteria}")
        
        # Layer 3: Filter
        filtered_ids = self.filter_engine.apply(candidate_movies, norm_criteria)
        print(f"[Layer 3] Filtered Count: {len(filtered_ids)} / {len(candidate_movies)}")
        
        # Construct Result
        # Create a simple justification for the top result if exists
        agent_rec = None
        if filtered_ids:
            top_id = filtered_ids[0]
            # Find title
            top_movie = next((m for m in candidate_movies if m['movieId'] == top_id), None)
            agent_rec = {
                "movieId": int(top_id),
                "title": top_movie['title'] if top_movie else "Unknown",
                "reason": f"Matches criteria: {norm_criteria}"
            }

        return {
            "filtered_movie_ids": filtered_ids,
            "filters_applied": norm_criteria,
            "agent_recommendation": agent_rec,
            "total_candidates": len(candidate_movies),
            "filtered_count": len(filtered_ids)
        }

# Singleton accessor
_agent_instance = None
def get_filter_agent() -> FilmFilterAgent:
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = FilmFilterAgent()
    return _agent_instance
