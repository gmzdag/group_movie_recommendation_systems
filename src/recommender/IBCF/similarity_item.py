"""
Item-Based Similarity Functions
-------------------------------
Computes item–item similarity matrices.
"""

import pandas as pd


def pearson_similarity(item_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Pearson similarity between items (movies).
    item_matrix: M × U matrix (movies as rows)

    Returns:
        similarity_df: M × M similarity matrix
    """
    # corr() already computes Pearson similarity
    sim = item_matrix.T.corr()

    # NaN → 0 similarity
    return sim.fillna(0)
