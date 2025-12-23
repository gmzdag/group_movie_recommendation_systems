import numpy as np
import pandas as pd

class UserBasedCF:
    def __init__(self, R, neighbors, user_means, item_means, global_mean, movies=None, K_PRED=None):
        self.R = R
        self.neighbors = neighbors
        self.user_means = user_means
        self.item_means = item_means
        self.global_mean = global_mean
        self.movies = movies
        self.K_PRED = K_PRED

    def predict(self, user_id, movie_id):
        # 1. Check if movie exists in training data
        if movie_id not in self.R.columns:
            return self.global_mean

        # 2. Get neighbors and their similarities
        user_neighbors = self.neighbors.get(user_id, {})
        if not user_neighbors:
            return self.global_mean

        # Apply K_PRED limit if specified
        if self.K_PRED is not None and len(user_neighbors) > self.K_PRED:
            # Neighbors are assumed to be sorted by similarity descending
            # We take the top K_PRED
            # Note: Since dicts preserve order, and neighbors were sorted when created, this works.
            # If not sure, we would need to sort again. But for performance we assume sorted input.
            import itertools
            user_neighbors = dict(itertools.islice(user_neighbors.items(), self.K_PRED))

        # Convert to arrays for vectorized op
        neighbor_ids = list(user_neighbors.keys())
        similarities = np.array(list(user_neighbors.values()))

        # 3. Get ratings of neighbors for this movie
        try:
             neighbor_ratings = self.R.loc[neighbor_ids, movie_id].values
        except KeyError:
             return self.global_mean

        # 4. Filter out NaN ratings (neighbors who haven't rated this movie)
        mask = ~np.isnan(neighbor_ratings)
        if not mask.any():
            # If no neighbors rated the movie, fall back to baseline
            # baseline_u = user_mean[u] + item_mean[i] - global_mean
            i_mean = self.item_means[movie_id]
            target_baseline = self.user_means.get(user_id, self.global_mean) + i_mean - self.global_mean
            return target_baseline

        valid_sims = similarities[mask]
        valid_ratings = neighbor_ratings[mask]
        valid_nids = np.array(neighbor_ids)[mask]

        # 5. Calculate prediction using Mean-Centered Approach (Standard Resnick)
        # This aligns better with Pearson correlation which measures linear correlation of deviations.
        # Formula: pred = mean_u + sum(sim_v * (r_v - mean_v)) / sum(|sim_v|)
        
        u_means_v = self.user_means.loc[valid_nids].values
        
        # Deviation of neighbor ratings from their own means
        deviations_v = valid_ratings - u_means_v
        
        num = np.sum(valid_sims * deviations_v)
        den = np.sum(np.abs(valid_sims))
        
        # Target user mean
        target_mean = self.user_means.get(user_id, self.global_mean)

        if den == 0:
            return target_mean
        
        return target_mean + (num / den)

    def predict_for_user(self, user_id, item_subset=None):
        """
        Vectorized prediction for one user on multiple items.
        Faster than calling predict() iteratively.
        """
        if user_id not in self.neighbors:
             return {}
             
        # Neighbors map: {neighbor_id: sim_score}
        n_map = self.neighbors[user_id]
        if not n_map:
            return {}
            
        n_ids = list(n_map.keys())
        sims = np.array(list(n_map.values()))
        
        # Get dense block for neighbors (Neighbors x AllItems)
        # Using self.R (pd.DataFrame). 
        # Slicing row-wise ok for typical frame.
        
        # If item_subset is provided, we filter columns.
        if item_subset is not None:
             # Filter items that exist in columns
             valid_items = [i for i in item_subset if i in self.R.columns]
             if not valid_items:
                 return {}
             n_ratings = self.R.loc[n_ids, valid_items].values # Shape: (K, M)
             items = valid_items
        else:
             n_ratings = self.R.loc[n_ids].values
             items = self.R.columns
             
        # Calculate deviations (r_v - mean_v)
        # mean_v shape: (K,)
        n_means = self.user_means.loc[n_ids].values[:, None] # (K, 1) to broadcast
        
        # Deviations (n_ratings has NaNs)
        deviations = n_ratings - n_means
        
        # Weighted Sum
        # sims shape: (K,)
        # score = sum(sim * dev) / sum(|sim|) ignoring NaNs
        
        # Weighted deviations: Broadcast sim to each column
        weighted_devs = sims[:, None] * deviations # Shape (K, M)
        
        # Numerator: Sum ignoring NaNs
        num = np.nansum(weighted_devs, axis=0) # Shape (M,)
        
        # Denominator: Sum of abs(sim) ONLY for rows where rating was present
        present_mask = ~np.isnan(n_ratings)
        abs_sims = np.abs(sims)[:, None] # (K, 1)
        den = np.sum(abs_sims * present_mask, axis=0) # Shape (M,)
        
        # Scores
        # Avoid div by zero
        with np.errstate(divide='ignore', invalid='ignore'):
             pred_devs = num / den
             
        # Add user mean
        u_mean = self.user_means.get(user_id, self.global_mean)
        preds = u_mean + pred_devs
        
        # Handle cases where den == 0 (no neighbors rated the item) -> use baseline or NaN
        # Standard fallback if no neighbors rated: Baseline? Or Global Mean?
        # Here we leave as NaN, filtered out later or filled. 
        # UBCF usually cannot predict if neighbors didn't see.
        
        return dict(zip(items, preds))

    def recommend(self, user_id, top_n=10):
        """
        Recommend top_n movies for a single user.
        """
        # Get all movies
        all_movies = self.R.columns
        
        # Identify watched movies
        if user_id in self.R.index:
            rated_movies = self.R.loc[user_id].dropna().index
        else:
            rated_movies = []
            
        # Candidates: movies not rated by user
        candidates = [m for m in all_movies if m not in rated_movies]
        
        # Predict score for each candidate
        # Optimization: We could matrix multiply, but predict is one-by-one currently.
        # For full recommendations usually we need faster approach, but for demo loop is ok.
        scores = []
        for mid in candidates:
            score = self.predict(user_id, mid)
            scores.append((mid, score))
            
        # Sort by score desc
        scores.sort(key=lambda x: x[1], reverse=True)
        top_scores = scores[:top_n]
        
        # Prepare dataframe
        results = []
        for mid, score in top_scores:
            title = f"Movie {mid}"
            if self.movies is not None and mid in self.movies.index:
                title = self.movies.loc[mid, 'title']
            
            results.append({
                'movieId': mid,
                'title': title,
                'score_raw': score,
                'score': min(5.0, max(0.5, score))  # Clip for display, but sorted by raw
            })
            
        return pd.DataFrame(results)

    def recommend_group(self, group_users, method='mean_score', top_n=10):
        """
        Recommend movies for a group of users.
        methods: mean_score, least_misery, most_pleasure
        """
        # Find candidates (union of unseen? or simply all movies?)
        # Usually we ignore movies that ANY user has seen? Or ALL users?
        # Let's say candidates are movies that NO ONE in the group has seen? 
        # Or simply all movies, and if someone saw it, we penalize?
        # Standard approach: exclude movies that *any* member has watched.
        
        all_movies = self.R.columns
        watched_set = set()
        for uid in group_users:
            if uid in self.R.index:
                watched = self.R.loc[uid].dropna().index
                watched_set.update(watched)
                
        candidates = [m for m in all_movies if m not in watched_set]
        
        group_scores = []
        
        for mid in candidates:
            member_preds = []
            for uid in group_users:
                pred = self.predict(uid, mid)
                member_preds.append(pred)
            
            if not member_preds:
                continue
                
            if method == 'mean_score':
                agg_score = np.mean(member_preds)
            elif method == 'least_misery':
                agg_score = np.min(member_preds)
            elif method == 'most_pleasure':
                agg_score = np.max(member_preds)
            else:
                agg_score = np.mean(member_preds)
                
            group_scores.append((mid, agg_score))
            
        group_scores.sort(key=lambda x: x[1], reverse=True)
        top_scores = group_scores[:top_n]
        
        results = []
        for mid, score in top_scores:
            title = f"Movie {mid}"
            if self.movies is not None and mid in self.movies.index:
                title = self.movies.loc[mid, 'title']
            
            results.append({
                'movieId': mid,
                'title': title,
                'score_raw': score, 
                'score': min(5.0, max(0.5, score)) # Clip for display
            })
            
        return pd.DataFrame(results)
