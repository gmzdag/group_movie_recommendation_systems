
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import time

def compute_neighbors_matrix(R_df, sim_metric='pearson', K=50, min_overlap=10, batch_size=1000):
    """
    Computes User-User neighbors using Vectorized Sparse Matrix operations.
    Massively faster than iterating pairs (Seconds vs Days).
    
    Args:
        R_df: User-Item DataFrame (Pandas) - NaNs for missing
        sim_metric: 'pearson', 'cosine', 'pearson_sw' (Significance Weighting)
        K: Top-K neighbors to keep per user
        min_overlap: Minimum intersection to consider valid
        batch_size: Number of users to process in one vector block (to save RAM)
        
    Returns:
        dict: {uid: {neighbor_uid: score, ...}}
    """
    print(f"[MATRIX] transforming to sparse matrix (Shape: {R_df.shape})...")
    start_total = time.time()
    
    # 0. Data Prep
    # Users and Items Mappings to indices
    user_ids = R_df.index.values
    item_ids = R_df.columns.values
    
    n_users = len(user_ids)
    n_items = len(item_ids)
    
    # Fill NaN with 0 for sparse conversion (we handle means separately for Pearson)
    R_filled = R_df.fillna(0)
    R_sparse = csr_matrix(R_filled.values)
    
    # Binary matrix for overlap counting: 1 if rated, 0 if not
    R_binary = (R_sparse > 0).astype(int)
    
    # 1. Pre-process for Pearson: Center the data
    # (u - u_bar) for rated items, 0 for unrated
    if 'pearson' in sim_metric:
        print("[MATRIX] Centering data for Pearson...")
        # Calculate row means (ignoring zeros/missing)
        # Sum of ratings / Count of ratings
        user_sums = np.array(R_sparse.sum(axis=1)).flatten()
        user_counts = np.array(R_binary.sum(axis=1)).flatten()
        user_means = np.divide(user_sums, user_counts, out=np.zeros_like(user_sums), where=user_counts!=0)
        
        # Subtract mean only from non-zero entries
        # We do this efficiently using COO format
        R_coo = R_sparse.tocoo()
        updates = R_coo.data - user_means[R_coo.row]
        
        # Rebuild centered sparse matrix
        R_centered = csr_matrix((updates, (R_coo.row, R_coo.col)), shape=R_sparse.shape)
        
        # For Cosine Similarity of Centered Data (=Pearson), we need L2 Norms
        # sklearn cosine_similarity does this, but we'll use it block-wise
        processing_matrix = R_centered
    else:
        # Pure Cosine
        processing_matrix = R_sparse

    # Result Dictionary
    all_neighbors = {}
    
    print(f"[MATRIX] Computing similarities in batches of {batch_size} users...")
    
    # 2. Block-wise Computation
    # We calculate Sim(Batch, All) = Batch @ All.T
    for start_idx in range(0, n_users, batch_size):
        end_idx = min(start_idx + batch_size, n_users)
        
        # Get batch of users
        batch_matrix = processing_matrix[start_idx:end_idx]
        
        # A. Similarity Scores (Cosine of Centered = Pearson)
        # dense output: (batch_size, n_users) - Careful with RAM here!
        # If n_users=76k, 1 row = 76k floats = ~600KB. 1000 rows = 600MB. Fine.
        sim_scores = cosine_similarity(batch_matrix, processing_matrix, dense_output=True)
        
        # B. Overlap Counts (if needed for SW or filtering)
        # Overlap = Binary_Batch @ Binary_All.T
        if min_overlap > 0 or 'sw' in sim_metric:
            batch_binary = R_binary[start_idx:end_idx]
            # This is dense dot product of sparse matrices
            overlap_counts = batch_binary.dot(R_binary.T).toarray()
            
            # Apply Min Overlap Filter
            mask_low_overlap = overlap_counts < min_overlap
            sim_scores[mask_low_overlap] = 0 # or -1
            
            # Apply Significance Weighting: sim * min(1, overlap/K_sw)
            if 'sw' in sim_metric:
                # Standard SW param usually K=50 or similar, here we assume it scales 
                # Let's use a fixed SW threshold, e.g., 50 (common literature default) 
                # or similar to min_overlap scaling. 
                # The user code had SW scaling factor 'K'. Let's default to standard or K param.
                # Note: Pearson SW formula in user code: sim * min(1, n/K_neighbors)
                # We use the K passed in args (neighbor count) as the SW damping factor too? 
                # Usually they are distinct parameters (Beta or shrinkage), but user code uses arg K.
                
                damping = np.minimum(1.0, overlap_counts / K)
                sim_scores = sim_scores * damping

        # C. Self-similarity removal
        # For each user in batch, set their own column index to -1
        # Row i in sim_scores corresponds to user (start_idx + i)
        for i in range(len(sim_scores)):
            global_idx = start_idx + i
            sim_scores[i, global_idx] = -9999
        
        # D. Top-K Extraction
        # argpartition is faster than sort
        # We want top K indices
        # If K > n_users, clip it
        eff_K = min(K, n_users - 1)
        
        # Get indices of top K
        # np.argpartition puts top K at the end
        top_k_indices = np.argpartition(sim_scores, -eff_K, axis=1)[:, -eff_K:]
        
        # For each user in batch, store results
        for i in range(len(sim_scores)):
            u_global_idx = start_idx + i
            u_id = user_ids[u_global_idx]
            
            # Sort the small top-K set to be sure (argpartition is unordered)
            # Get values
            relevant_indices = top_k_indices[i]
            relevant_scores = sim_scores[i, relevant_indices]
            
            # Sort descending
            sorted_order = np.argsort(relevant_scores)[::-1]
            final_indices = relevant_indices[sorted_order]
            final_scores = relevant_scores[sorted_order]
            
            # Filter non-positive sims if desired (UserCF usually positive only)
            valid_mask = final_scores > 0
            final_indices = final_indices[valid_mask]
            final_scores = final_scores[valid_mask]
            
            # Store map {neighbor_id: score}
            # Map indices back to UserIDs
            u_neighbors = {}
            for rank, idx in enumerate(final_indices):
                neighbor_id = user_ids[idx]
                score = float(final_scores[rank])
                u_neighbors[neighbor_id] = score
                
            all_neighbors[u_id] = u_neighbors
        
        # Progress Log
        if (end_idx // 1000) % 5 == 0 or end_idx == n_users:
            print(f"[MATRIX] Processed {end_idx}/{n_users} users ({(end_idx/n_users)*100:.1f}%)")

    print(f"[MATRIX] Completed in {time.time() - start_total:.1f}s")
    return all_neighbors

