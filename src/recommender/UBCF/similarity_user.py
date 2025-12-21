import numpy as np
from scipy.spatial.distance import cosine

def pearson_sw(u, v, MIN_OVERLAP=10, K=20):
    both = u.dropna().index.intersection(v.dropna().index)
    n = len(both)
    if n < MIN_OVERLAP:
        return np.nan
    
    u_mc = u[both] - u[both].mean()
    v_mc = v[both] - v[both].mean()

    num = (u_mc * v_mc).sum()
    den = np.sqrt((u_mc**2).sum()) * np.sqrt((v_mc**2).sum())
    if den == 0:
        return np.nan
    
    return float((num / den) * min(1, n / K))


def pearson_shrink(u, v, MIN_OVERLAP=10, LAMBDA=20):
    both = u.dropna().index.intersection(v.dropna().index)
    n = len(both)
    if n < MIN_OVERLAP:
        return np.nan

    u_mc = u[both] - u[both].mean()
    v_mc = v[both] - v[both].mean()

    num = (u_mc * v_mc).sum()
    den = np.sqrt((u_mc**2).sum()) * np.sqrt((v_mc**2).sum())
    if den == 0:
        return np.nan

    r = num / den
    return float((n * r) / (n + LAMBDA))


def cosine_sim(u, v, MIN_OVERLAP=10):
    both = u.dropna().index.intersection(v.dropna().index)
    if len(both) < MIN_OVERLAP:
        return np.nan

    u_vec = u[both].values
    v_vec = v[both].values
    
    # Check for zero vectors to avoid division by zero in cosine
    if (u_vec == 0).all() or (v_vec == 0).all():
        return np.nan

    # scipy.spatial.distance.cosine returns 1 - cosine_similarity
    return float(1.0 - cosine(u_vec, v_vec))


def spearman_rank(u, v, MIN_OVERLAP=2):
    from scipy.stats import spearmanr
    both = u.dropna().index.intersection(v.dropna().index)
    if len(both) < MIN_OVERLAP:
        return np.nan

    u_vec = u[both].values
    v_vec = v[both].values
    
    # Check for constant vectors to avoid warnings/NaNs
    if np.all(u_vec == u_vec[0]) or np.all(v_vec == v_vec[0]):
         return np.nan
    
    corr, _ = spearmanr(u_vec, v_vec)
    return float(corr)

def spearman_sw(u, v, MIN_OVERLAP=10, K=20):
    """
    Spearman Rank Correlation with Significance Weighting.
    Penalizes correlations based on small number of overlapping items.
    """
    from scipy.stats import spearmanr
    both = u.dropna().index.intersection(v.dropna().index)
    n = len(both)
    if n < MIN_OVERLAP:
        return np.nan

    u_vec = u[both].values
    v_vec = v[both].values
    
    # Check for constant vectors
    if np.all(u_vec == u_vec[0]) or np.all(v_vec == v_vec[0]):
         return np.nan

    corr, _ = spearmanr(u_vec, v_vec)
    
    if np.isnan(corr):
        return np.nan

    # Significance Weighting: multiply by min(1, n/K)
    # This reduces confidence in correlations found with few overlapping items
    return float(corr * min(1, n / K))
