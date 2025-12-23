
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Sequence, Set, Tuple
from sklearn.metrics.pairwise import cosine_similarity

from recommender.CB.content_based import ContentBasedModel


class WatchlistRecommender:
    """
    Watchlist-based Content Recommender (Individual & Group).
    
    SCIENTIFIC APPROACH:
    - Uses ContentBasedModel for TF-IDF vectorization
    - Individual: AVERAGE profile strategy (mean of all watchlist items)
    - Group: Consensus scoring with disagreement penalty
    
    This is different from Hybrid Model 3 which uses MAX similarity for individuals.
    """
    def __init__(self, movies_df: pd.DataFrame, ratings_df: pd.DataFrame, 
                 watchlist_df: pd.DataFrame, disagreement_penalty: float = 0.5,
                 cb_model: Optional['ContentBasedModel'] = None):
        """
        Args:
            movies_df: Movies metadata
            ratings_df: User ratings (for ContentBasedModel)
            watchlist_df: User watchlists
            disagreement_penalty: Weight for disagreement penalty in consensus scoring (0.0-1.0)
                                 Higher = more penalty for disagreement (favors fairness)
                                 Lower = less penalty (favors average similarity)
            cb_model: Optional pre-built ContentBasedModel (for optimization speed)
        """
        self.movies_df = movies_df
        self.ratings_df = ratings_df
        self.watchlist_df = watchlist_df
        self.disagreement_penalty = disagreement_penalty
        
        # Prepare Title Map
        self.title_map = (
            self.movies_df[["movieId", "title"]]
            .dropna(subset=["movieId"])
            .assign(movieId=lambda df: df["movieId"].astype(int))
            .set_index("movieId")["title"]
            .fillna("Unknown Title")
            .to_dict()
        )
        
        # Initialize or reuse Content-Based Model (for TF-IDF and vectors)
        if cb_model is not None:
            print(f"[DEBUG] Reusing pre-built ContentBasedModel (optimization mode)")
            self.cb_model = cb_model
        else:
            self.cb_model = ContentBasedModel(self.movies_df, self.ratings_df)

    def get_user_seeds(self, user_id: int) -> Set[int]:
        """Returns movie IDs from a user's watchlist."""
        return set(
            self.watchlist_df[self.watchlist_df["userId"] == user_id]["movieId"]
            .astype(int)
            .tolist()
        )

    def build_profile_vector(self, seed_ids: Sequence[int]) -> Optional[np.ndarray]:
        """Creates an average profile vector from seed movie IDs."""
        indices = [self.cb_model.movie_to_idx[mid] for mid in seed_ids if mid in self.cb_model.movie_to_idx]
        if not indices:
            return None

        # (N, Features) -> mean -> (1, Features)
        profile_sparse = self.cb_model.tfidf_matrix[indices].mean(axis=0)
        profile = np.asarray(profile_sparse).reshape(1, -1)
        return profile

    def get_candidate_movies(self, group_users: Sequence[int]) -> Set[int]:
        """
        Returns candidate movies: All movies minus (Group Watchlist + Group Rated).
        """
        group_watchlisted = set(
            self.watchlist_df[self.watchlist_df["userId"].isin(group_users)]["movieId"].astype(int)
        )
        rated = set(
            self.ratings_df[self.ratings_df["userId"].isin(group_users)]["movieId"].astype(int)
        )
        excluded = group_watchlisted | rated
        all_movie_ids = set(self.title_map.keys())
        return all_movie_ids - excluded

    def predict(self, user_id: int, movie_id: int) -> float:
        """
        Individual prediction using AVERAGE profile strategy.
        
        SCIENTIFIC NOTE:
        - Builds average profile from ALL watchlist items
        - Different from Hybrid Model 3's MAX similarity approach
        
        Returns: 0.0 to 5.0 (cosine similarity * 5)
        """
        seeds = self.get_user_seeds(user_id)
        if not seeds or movie_id not in self.cb_model.movie_to_idx:
            return np.nan
        
        profile = self.build_profile_vector(list(seeds))
        if profile is None:
            return np.nan
        
        idx = self.cb_model.movie_to_idx[movie_id]
        movie_vec = self.cb_model.tfidf_matrix[idx]
        score = cosine_similarity(profile, movie_vec)[0][0]
        
        return score * 5.0

    def recommend(self, group_users: Sequence[int], candidates: Optional[List[int]] = None, top_k: int = 10) -> Dict:
        """
        Generates group recommendations.
        
        Args:
            group_users: List of user IDs in the group
            candidates: Optional list of candidate movie IDs. If None, will be computed automatically.
            top_k: Number of recommendations to return
        """
        # 1. Build User Profiles
        user_profiles = {}
        user_seeds_map = {}
        
        for uid in group_users:
            seeds = self.get_user_seeds(uid)
            user_seeds_map[uid] = seeds
            if not seeds:
                print(f"[WARN] User {uid} has empty watchlist, skipping profile.")
                continue
                
            profile = self.build_profile_vector(list(seeds))
            if profile is not None:
                user_profiles[uid] = profile
        
        if not user_profiles:
            raise ValueError("No valid profiles could be built for the group.")

        # 2. Identify Common Watchlist Items (Guaranteed Picks)
        all_seeds = [s for s in user_seeds_map.values() if s]
        common = set.intersection(*all_seeds) if all_seeds else set()

        # 3. Get Candidates
        if candidates is None:
            candidates = self.get_candidate_movies(group_users)
        else:
            # Use provided candidates (for evaluation consistency)
            candidates = list(candidates)

        # 4. Score Candidates
        final_scores: List[Tuple[int, float, bool]] = []
        
        # Prepare Common Profile for Boosting
        common_profile_vec = None
        if common:
            print(f"[INFO] Common {len(common)} watchlisted movies found. Common boost active.")
            common_profile_vec = self.build_profile_vector(list(common))

        for mid in candidates:
            idx = self.cb_model.movie_to_idx.get(mid)
            if idx is None:
                continue
                
            movie_vec = self.cb_model.tfidf_matrix[idx]
            
            # Calculate score for each user
            metrics = []
            for uid, profile in user_profiles.items():
                sim = cosine_similarity(profile, movie_vec)[0][0]
                metrics.append(float(sim))
            
            if not metrics:
                continue

            avg_score = np.mean(metrics)
            disagreement = np.std(metrics)
            
            # Consensus Score: Average - Penalty for Disagreement
            # Use tunable disagreement_penalty parameter
            consensus_score = avg_score - (disagreement * self.disagreement_penalty)

            # Common Boost
            is_common_boosted = False
            if common_profile_vec is not None:
                 common_sim = cosine_similarity(common_profile_vec, movie_vec)[0][0]
                 if common_sim > avg_score:
                     consensus_score *= 1.2
                     is_common_boosted = True
            
            final_scores.append((mid, consensus_score, is_common_boosted))

        # 5. Rank
        ranked = sorted(final_scores, key=lambda x: x[1], reverse=True)
        top_items = ranked[:top_k]

        # 6. Generate Explanations
        explained_results = []
        search_candidates = list(common) if common else []
        all_seeds_list = [m for s in user_seeds_map.values() for m in s]

        for mid, score, is_boosted in top_items:
            idx = self.cb_model.movie_to_idx.get(mid)
            movie_vec = self.cb_model.tfidf_matrix[idx]
            
            # Find best reference movie for explanation
            best_ref_sim = -1.0
            best_ref_title = "?"
            
            def find_best_match(ref_ids):
                b_sim, b_title = -1.0, "?"
                for ref_id in ref_ids:
                    r_idx = self.cb_model.movie_to_idx.get(ref_id)
                    if r_idx is None: 
                        continue
                    r_vec = self.cb_model.tfidf_matrix[r_idx]
                    sim = cosine_similarity(r_vec, movie_vec)[0][0]
                    if sim > b_sim:
                        b_sim = sim
                        b_title = self.title_map.get(ref_id, str(ref_id))
                return b_sim, b_title

            # Check common picks first
            if search_candidates:
                best_ref_sim, best_ref_title = find_best_match(search_candidates)
            
            # If low similarity, check all individual watchlists
            if best_ref_sim < 0.05:
                g_sim, g_title = find_best_match(all_seeds_list)
                if g_sim > best_ref_sim:
                    best_ref_sim = g_sim
                    best_ref_title = g_title

            reason = "High Consensus"
            if best_ref_sim > 0.05:
                 reason = f"Similar to: {best_ref_title}"
            
            if is_boosted and "Similar to:" not in reason:
                reason = "Matches Mutual Taste"
                
            explained_results.append((mid, score, reason))

        return {
            "guaranteed_picks": sorted(common),
            "recommended_movies": explained_results,
            "movie_titles": self.title_map,
        }
    
    def recommend_for_group(self, user_ids: List[int], candidates: Optional[List[int]] = None, 
                           top_k: int = 10) -> Dict:
        """
        API compatibility wrapper for recommend method.
        Allows WatchlistRecommender to work with optimization scripts.
        """
        return self.recommend(user_ids, candidates, top_k)
