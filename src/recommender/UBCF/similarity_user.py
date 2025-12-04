import numpy as np

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
