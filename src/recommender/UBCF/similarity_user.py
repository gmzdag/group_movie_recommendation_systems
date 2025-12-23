import numpy as np
from scipy.spatial.distance import cosine

def pearson_sw(u, v, MIN_OVERLAP=10, K=20):
    """
    Pearson Correlation with Significance Weighting.
    
    Uses mean-centering normalization as per Resnick et al. (1994) GroupLens.
    Each user's ratings are centered around their own mean to account for 
    individual rating biases (some users rate higher/lower on average).
    
    Significance weighting (Herlocker et al., 1999) penalizes correlations
    based on few overlapping items: correlation * min(1, n/K)
    
    References:
    - Resnick et al. (1994): GroupLens collaborative filtering
    - Herlocker et al. (1999): An algorithmic framework for CF
    """
    both = u.dropna().index.intersection(v.dropna().index)
    n = len(both)
    if n < MIN_OVERLAP:
        return np.nan
    
    # Mean-centering: Subtract each user's own mean (Resnick et al., 1994)
    # This accounts for individual rating biases
    u_mc = u[both] - u[both].mean()
    v_mc = v[both] - v[both].mean()

    num = (u_mc * v_mc).sum()
    den = np.sqrt((u_mc**2).sum()) * np.sqrt((v_mc**2).sum())
    if den == 0:
        return np.nan
    
    # Significance weighting (Herlocker et al., 1999)
    return float((num / den) * min(1, n / K))


def pearson_shrink(u, v, MIN_OVERLAP=10, LAMBDA=20):
    """
    Pearson Correlation with Shrinkage (Bayesian approach).
    
    Uses mean-centering normalization (Resnick et al., 1994) and applies
    shrinkage towards zero for correlations based on few items.
    Formula: (n * r) / (n + LAMBDA)
    
    This is a Bayesian approach that assumes prior correlation of 0,
    and shrinks the observed correlation towards this prior.
    
    References:
    - Resnick et al. (1994): GroupLens collaborative filtering
    - Bell & Koren (2007): Scalable Collaborative Filtering
    """
    both = u.dropna().index.intersection(v.dropna().index)
    n = len(both)
    if n < MIN_OVERLAP:
        return np.nan

    # Mean-centering: Subtract each user's own mean (Resnick et al., 1994)
    u_mc = u[both] - u[both].mean()
    v_mc = v[both] - v[both].mean()

    num = (u_mc * v_mc).sum()
    den = np.sqrt((u_mc**2).sum()) * np.sqrt((v_mc**2).sum())
    if den == 0:
        return np.nan

    r = num / den
    # Shrinkage: penalize correlations based on few overlaps
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
