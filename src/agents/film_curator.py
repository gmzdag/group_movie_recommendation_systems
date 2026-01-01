"""
Film Filter Agent - Smart Logic
-------------------------------------------
A deterministic, 3-layer architecture for filtering movies based on user intent.
Layers:
1. LLMFilterExtractor: Intelligent extraction, translation, and expansion of user intent.
2. NormalizationLayer: Basic validation (previously managed maps, now delegated to LLM).
3. DeterministicMovieFilter: Evidence-based filtering against metadata and text.

Design guarantees reproducibility while leveraging LLM intelligence for translation 
and synonym expansion.
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
# LAYER 1: LLM INTENT EXTRACTOR (SMART)
# ============================================================================

class LLMFilterExtractor:
    """
    Layer 1: Intelligent Intent Extraction.
    Uses LLM to:
    1. Translate user prompt to English.
    2. Map to standard TMDB Genres.
    3. Generate 'Bag of Words' (Synonyms/Related) for themes.
    """
    
    def __init__(self, model):
        self.model = model
        
    def extract(self, user_prompt: str) -> Dict[str, Any]:
        """
        Extract structured constraints from user prompt.
        """
        system_prompt = """
You are a smart movie recommendation assistant.
Your GOAL: Translate the user's intent into a structured SEARCH QUERY.

INPUT: User search query (in ANY language).
OUTPUT: Strict JSON object.

TASKS:
1. TRANSLATE: Convert all concepts to English (e.g. "Korku" -> "Horror").
2. STANDARDIZE: Map genres to valid TMDB genres:
   [Action, Adventure, Animation, Comedy, Crime, Documentary, Drama, Family, Fantasy, History, Horror, Music, Mystery, Romance, Science Fiction, TV Movie, Thriller, War, Western].
3. EXPAND: For 'themes', generate keywords ONLY for specific topics/subjects.
   - INCLUDE VARIATIONS: Plurals, adjectives, related forms (e.g. "alien" -> "aliens", "alien invasion").
   - DO NOT generate themes if the request is covered entirely by standard Genres.
   - DO NOT infer genres from concepts. If user says "Teen movies", Genres should be NULL (not Drama/Comedy).
   - Example : "Horror movies" -> Genres: ["Horror"], Themes: null.
   - Example : "Komedi" -> Genres: ["Comedy"], Themes: null.
   - Example : "Teen movies" -> Genres: null, Themes: ["teen", "high school", "coming of age"].
   - Example : "Space Horror" -> Genres: ["Horror", "Science Fiction"], Themes: ["space", "spaceship", "alien"].

SCHEMA:
{
  "genres": [string] | null,          // Standard TMDB Genres only. ONLY if explicitly requested.
  "mood": string | null,              // e.g., "dark", "light", "intense", "sad"
  "themes": [string] | null,          // List of related English keywords (synonyms)
  "actors": string | null,            // Name of actor if mentioned
  "director": string | null,          // Name of director if mentioned
  "year_min": integer | null,
  "year_max": integer | null,
  "country": string | null,           // Standard English country name (e.g. "France")
  "min_runtime": integer | null,
  "max_runtime": integer | null
}

EXAMPLES:
Input: "Komik bir şeyler aç"
Output: {"genres": ["Comedy"], "mood": "light", "themes": ["funny", "hilarious", "laugh", "happy"]}

Input: "Fransız sanat filmi"
Output: {"country": "France", "genres": ["Drama"], "themes": ["art house", "philosophical", "french cinema", "cinematic"]}

Input: "90larda geçen uzay filmi"
Output: {"themes": ["space", "sci-fi", "aliens", "universe"], "year_min": 1990, "year_max": 1999}

Input: "Teen movies"
Output: {"genres": null, "themes": ["teen", "high school", "coming of age"]}
"""
        
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
                    # --- SAFEGUARD: PURE GENRE CHECK ---
                    # Prevents the LLM from hallucinating themes for queries that are strictly genres.
                    # Example: "Comedy" -> Should not generate themes=["funny"] to ensure broad recall.
                    # We strictly set themes=None if the prompt matches a recognized genre name.
                    clean_prompt = user_prompt.strip().lower()
                    pure_genres = {
                        'komedi', 'comedy', 'korku', 'horror', 'aksiyon', 'action', 
                        'dram', 'drama', 'bilim kurgu', 'sci-fi', 'science fiction',
                        'macera', 'adventure', 'romantik', 'romance', 'aşk', 
                        'animasyon', 'animation', 'belgesel', 'documentary',
                        'suç', 'crime', 'gizem', 'mystery'
                    }
                    
                    is_pure = False
                    if clean_prompt in pure_genres:
                        is_pure = True
                    else:
                        for g in pure_genres:
                            if clean_prompt == f"{g} movies" or clean_prompt == f"{g} filmi" or clean_prompt == f"{g} filmleri":
                                is_pure = True
                                break
                                
                    if is_pure and parsed.get('genres'):
                        # Only apply safeguards if genres were actually detected
                        print(f"[LLMFilterExtractor] Detected Pure Genre request ('{clean_prompt}'). Wiping generated themes.")
                        parsed['themes'] = None

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
                # Last ditch: try simple parsing if it's just keys
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
# LAYER 2: VALIDATION / NORMALIZATION (Simplified)
# ============================================================================

class NormalizationLayer:
    """
    Layer 2: Hybrid Normalization.
    Combines LLM intelligence with deterministic synonym expansion for max recall.
    """
    
    def __init__(self):
        # Dictionary for deterministic expansion of common themes
        # This helps because LLM often misses plurals/variations (e.g. "teen" vs "teens")
        self.theme_map = {
            'teen': ['teen', 'teens', 'teenager', 'teenagers', 'teenage', 'adolescent', 'adolescence', 'high school', 'youth', 'young adult', 'coming of age'],
            'teenager': ['teen', 'teens', 'teenager', 'teenagers', 'teenage', 'high school'],
            'high school': ['high school', 'student', 'students', 'campus', 'teen', 'teens'],
            'space': ['space', 'spaceship', 'universe', 'galaxy', 'planet', 'alien', 'aliens', 'astronaut', 'cosmos', 'star wars', 'star trek'],
            'alien': ['alien', 'aliens', 'extraterrestrial', 'ufo', 'martian', 'creature'],
            'zombie': ['zombie', 'zombies', 'undead', 'virus', 'infection', 'apocalypse', 'survival'],
            'love': ['love', 'romance', 'relationship', 'couple', 'marriage', 'dating', 'heartbreak'],
            'war': ['war', 'battle', 'soldier', 'soldiers', 'military', 'army', 'combat', 'wwii', 'vietnam'],
            'car': ['car', 'cars', 'racing', 'race', 'driver', 'vehicle', 'automotive', 'speed'],
            'magic': ['magic', 'wizard', 'witch', 'spell', 'fantasy', 'supernatural', 'magical'],
            'christmas': ['christmas', 'xmas', 'holiday', 'santa', 'noel', 'festive', 'winter'],
            'new year': ['new year', 'new years', 'holiday'],
        }

    def normalize(self, criteria: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize and EXPAND criteria."""
        normalized = criteria.copy()
        
        # Ensure Genres is a list
        if normalized.get('genres'):
            if isinstance(normalized['genres'], str):
                normalized['genres'] = [normalized['genres']]
            normalized['genres'] = [str(g).title() for g in normalized['genres']]

        # Ensure Themes is a list AND EXPAND IT
        if normalized.get('themes'):
            raw_themes = normalized['themes']
            if isinstance(raw_themes, str): raw_themes = [raw_themes]
            
            expanded_themes = set()
            for t in raw_themes:
                t_key = str(t).lower().strip()
                
                # 1. Add the original term (cleaned)
                expanded_themes.add(t_key)
                
                # 2. Check map for exact match expansion
                if t_key in self.theme_map:
                    expanded_themes.update(self.theme_map[t_key])
                
                # 3. Simple heuristic checks (if not in map)
                # Ensure plurals are covered for basic words if we didn't map them
                if not t_key in self.theme_map:
                    if not t_key.endswith('s'):
                        expanded_themes.add(t_key + 's') # dog -> dogs
            
            normalized['themes'] = list(expanded_themes)
            
        # Ensure Country is string
        if normalized.get('country') and not isinstance(normalized['country'], str):
             if isinstance(normalized['country'], list) and len(normalized['country']) > 0:
                 normalized['country'] = str(normalized['country'][0])

        return normalized


# ============================================================================
# LAYER 3: DETERMINISTIC MOVIE FILTER (FLEXIBLE)
# ============================================================================

class DeterministicMovieFilter:
    """
    Layer 3: Evidence-Based Filtering.
    Checks metadata columns and text fields (Overview, Keywords, Title).
    Uses 'ANY' logic for themes/keywords to allow broad finding.
    """
    
    def apply(self, movies: List[Dict], criteria: Dict[str, Any]) -> List[int]:
        """
        Filter movies based on criteria.
        Returns list of matching movie IDs.
        """
        filtered_ids = []
        
        # Pre-process criteria for speed
        c_country = criteria.get('country')
        c_year_min = criteria.get('year_min')
        c_year_max = criteria.get('year_max')
        c_genres = criteria.get('genres')
        # Themes is now a list of related keywords (OR logic)
        c_themes = criteria.get('themes')
        
        c_mood = criteria.get('mood')
        c_actors = criteria.get('actors')
        c_director = criteria.get('director')
        
        # Mood Keywords Registry (Fallback/Expansion)
        mood_keywords = {
            'dark': ['dark', 'noir', 'grim', 'horror', 'thriller', 'bleak', 'crime'],
            'light': ['light', 'funny', 'comedy', 'feel-good', 'cheerful', 'happy'],
            'intense': ['intense', 'action', 'thriller', 'suspense', 'fast', 'adrenaline'],
            'calm': ['calm', 'drama', 'slow', 'quiet', 'peaceful', 'philosophical'],
            'sad': ['sad', 'drama', 'tragic', 'cry', 'emotional', 'melancholy']
        }
        
        target_mood_words = []
        if c_mood:
            key = c_mood.lower()
            if key in mood_keywords:
                target_mood_words = mood_keywords[key]
            else:
                target_mood_words = [key]

        for m in movies:
            passes = True
            
            # --- PREPARE DATA ---
            m_title = str(m.get('title', '')).lower()
            m_overview = str(m.get('overview', '')).lower()
            m_keywords = str(m.get('keywords', '')).lower()
            m_countries = str(m.get('countries', '')).lower() 
            m_year = m.get('year', 0)
            m_genres = [str(g).lower() for g in m.get('genres', [])]
            m_actors = str(m.get('actors', '')).lower()
            m_director = str(m.get('director', '')).lower()
            
            # Include genres in text blob so filtering by mood (which maps to genres) works
            # e.g. mood='light' -> keyword 'comedy' -> matches Genre='Comedy'
            m_text_blob = f"{m_title} {m_overview} {m_keywords} {' '.join(m_genres)}"
            
            # --- APPLY FILTERS ---

            # 1. Year (Strict)
            if c_year_min is not None and passes:
                if m_year < c_year_min: passes = False
            if c_year_max is not None and passes:
                if m_year > c_year_max: passes = False
                
            # 2. Country (Text Evidence)
            if c_country and passes:
                tgt = c_country.lower()
                evidence_found = (tgt in m_countries) or (tgt in m_text_blob)
                if not evidence_found:
                    passes = False
            
            # 3. Genres (ALL Logic)
            # Enforce strict intersection logic for genres to support requests like "Action Comedy".
            # The movie must match ALL requested genres.
            # Inference issues (e.g., "Teen" -> "Adventure, Comedy") are handled by the LLM prompt instructions.
            if c_genres and passes:
                for g in c_genres:
                    tgt = g.lower()
                    
                    # Direct genre match
                    genre_match = False
                    if any(tgt == mg or tgt in mg for mg in m_genres):
                        genre_match = True
                    # Fallback text match (e.g. "sci-fi" in keywords)
                    elif tgt in m_text_blob: 
                        genre_match = True
                    
                    if not genre_match:
                        passes = False
                        break
                
            # 4. Themes (Smart OR Logic)
            # LLM gives us ["space", "aliens", "universe"] for "Space movies".
            # We want to match if ANY of these appear.
            if c_themes and passes:
                match_found = False
                for t in c_themes:
                    # Use regex word boundary check for precision
                    # Escape the term just in case it has special chars
                    pattern = r'\b' + re.escape(t.lower()) + r'\b'
                    if re.search(pattern, m_text_blob):
                        match_found = True
                        break
                
                if not match_found:
                    passes = False
                
            # 5. Actors (Strict-ish)
            if c_actors and passes:
                if c_actors.lower() not in m_actors and c_actors.lower() not in m_text_blob:
                    passes = False

            # 6. Director (Strict-ish)
            if c_director and passes:
                if c_director.lower() not in m_director and c_director.lower() not in m_text_blob:
                    passes = False

            # 7. Mood (Keyword Match)
            if c_mood and passes:
                # Match ANY mood keyword
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
    Orchestrates the Smart filtering process.
    """
    
    def __init__(self):
        self.model = self._setup_model()
        
        # Initialize Layers
        self.extractor = LLMFilterExtractor(self.model)
        self.normalizer = NormalizationLayer()
        self.filter_engine = DeterministicMovieFilter()
        
        # Load Enrichment Data (Keywords, Overview, Credits)
        self.metadata_map = self._load_metadata()

    def _load_metadata(self):
        """Load TMDB metadata for content-based filtering."""
        try:
            import pandas as pd
            df = pd.read_csv('data/movies_tmdb.csv')
            # Create a dict for fast O(1) lookups: movieId -> dict of attributes
            # Ensure movieId is int for matching
            df['movieId'] = pd.to_numeric(df['movieId'], errors='coerce')
            df = df.dropna(subset=['movieId'])
            df['movieId'] = df['movieId'].astype(int)
            
            # Select relevant columns
            cols = ['movieId', 'Keywords', 'Overview', 'Director', 'Actors', 'Production_Countries']
            # Only keep cols that exist
            cols = [c for c in cols if c in df.columns]
            
            # Convert to dict
            meta_dict = df[cols].set_index('movieId').to_dict('index')
            print(f"[FilmFilterAgent] Loaded metadata for {len(meta_dict)} movies.")
            return meta_dict
        except Exception as e:
            print(f"[FilmFilterAgent] Warning: Could not load metadata: {e}")
            return {}

    def _setup_model(self):
        """Configure the LLM model (LiteLLM or HF Inference)."""
        gemini_key = get_api_key('gemini')
        hf_token = get_api_key('huggingface')
        
        # Option 1: Gemini via LiteLLM (Preferred for 'Smart' Logic)
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
        Execute the pipeline.
        """
        print(f"[FilmFilterAgent] Processing prompt: '{user_prompt}'")
        
        # Enrich candidates with Metadata BEFORE filtering
        # This allows us to search Overview, Keywords, Director etc.
        enriched_count = 0
        for m in candidate_movies:
            mid = m.get('movieId')
            if mid in self.metadata_map:
                meta = self.metadata_map[mid]
                # Update inplace
                m['keywords'] = str(meta.get('Keywords', ''))
                m['overview'] = str(meta.get('Overview', ''))
                m['director'] = str(meta.get('Director', ''))
                m['actors'] = str(meta.get('Actors', ''))
                m['countries'] = str(meta.get('Production_Countries', ''))
                enriched_count += 1
                
        print(f"[FilmFilterAgent] Enriched {enriched_count} / {len(candidate_movies)} candidates with Metadata.")
        
        # Layer 1: Extract Intent (Smart)
        raw_criteria = self.extractor.extract(user_prompt)
        print(f"[Layer 1] Extracted: {raw_criteria}")
        
        # Layer 2: Normalize (Basic Type Check)
        norm_criteria = self.normalizer.normalize(raw_criteria)
        print(f"[Layer 2] Normalized: {norm_criteria}")
        
        # Layer 3: Filter (Flexible Evidence)
        filtered_ids = self.filter_engine.apply(candidate_movies, norm_criteria)
        print(f"[Layer 3] Filtered Count: {len(filtered_ids)} / {len(candidate_movies)}")
        
        # Construct Result
        agent_rec = None
        if filtered_ids:
            top_id = filtered_ids[0]
            top_movie = next((m for m in candidate_movies if m['movieId'] == top_id), None)
            agent_rec = {
                "movieId": int(top_id),
                "title": top_movie['title'] if top_movie else "Unknown",
                "reason": f"Matches intent: {norm_criteria}"
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
