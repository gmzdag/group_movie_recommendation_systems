import numpy as np
from scipy.spatial.distance import cosine

def pearson_sw(u, v, MIN_OVERLAP=2, K=20):
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
    
    return (num / den) * min(1, n / K)


def pearson_shrink(u, v, MIN_OVERLAP=2, LAMBDA=10):
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
    return (n * r) / (n + LAMBDA)


def cosine_sim(u, v, MIN_OVERLAP=2):
    both = u.dropna().index.intersection(v.dropna().index)
    if len(both) < MIN_OVERLAP:
        return np.nan

    u_vec = u[both].values
    v_vec = v[both].values
    
    # Check for zero vectors to avoid division by zero in cosine
    if (u_vec == 0).all() or (v_vec == 0).all():
        return np.nan

    # scipy.spatial.distance.cosine returns 1 - cosine_similarity
    return 1.0 - cosine(u_vec, v_vec)
