"""
Hyperparameter Experiment for Item-Based Collaborative Filtering
with Prediction Formula (Bias Reconstruction)

This script:
1. Tests combinations of:
   - normalization: raw, mean_center, zscore
   - similarity: cosine, pearson
   - min_ratings: 3, 5, 10
   - top_k: 10, 20, 40, 60
2. Evaluates using MAE + RMSE
3. Saves results to itemcf_results.csv
4. Draws:
   - RMSE vs top_k
   - Top 10 best configurations (bar chart)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity

from src.recommender.data_loader import load_ratings, build_user_movie_matrix


# ------------------------------------------------------------
# Normalization Methods
# ------------------------------------------------------------
def normalize_raw(mat):
    return mat

def normalize_mean_center(mat):
    return mat.sub(mat.mean(axis=1), axis=0).fillna(0)

def normalize_zscore(mat):
    mean = mat.mean(axis=1)
    std = mat.std(axis=1).replace(0, 1)
    return mat.sub(mean, axis=0).div(std, axis=0).fillna(0)

normalizers = {
    "raw": normalize_raw,
    "mean_center": normalize_mean_center,
    "zscore": normalize_zscore
}


# ------------------------------------------------------------
# Pearson Similarity
# ------------------------------------------------------------
def pearson_sim(mat_T):
    return mat_T.T.corr().fillna(0).values


# ------------------------------------------------------------
#  ItemCF Prediction Formula
# ------------------------------------------------------------
def predict_rating(user_original, user_centered, movie_id, item_sim, k):
    """
    user_original: raw ratings (0 = missing)
    user_centered: normalized ratings (centered or zscore)
    """

    sims = item_sim[movie_id].drop(movie_id)
    top_k = sims.sort_values(ascending=False).head(k)

    neigh_ratings = user_centered[top_k.index]

    # If user has no neighbors ratings, fallback
    if np.all(neigh_ratings == 0):
        return user_original[user_original > 0].mean()

    weighted_sum = np.dot(top_k.values, neigh_ratings) / np.sum(np.abs(top_k.values))

    user_mean = user_original[user_original > 0].mean()
    return user_mean + weighted_sum


# ------------------------------------------------------------
# Evaluation Function
# ------------------------------------------------------------
def evaluate_itemcf(norm, sim_type, min_ratings, k):
    ratings = load_ratings()

    # Filter movies by minimum count
    counts = ratings["movieId"].value_counts()
    valid_ids = counts[counts >= min_ratings].index
    ratings = ratings[ratings["movieId"].isin(valid_ids)]

    raw_um = build_user_movie_matrix(ratings)
    norm_um = normalizers[norm](raw_um)

    # Similarity matrix
    if sim_type == "cosine":
        item_sim = pd.DataFrame(
            cosine_similarity(norm_um.T),
            index=norm_um.columns,
            columns=norm_um.columns
        )
    else:
        item_sim = pd.DataFrame(
            pearson_sim(norm_um.T),
            index=norm_um.columns,
            columns=norm_um.columns
        )

    # Test sample
    test = ratings.sample(min(1000, len(ratings)))
    y_true, y_pred = [], []

    for _, row in test.iterrows():
        user = row["userId"]
        movie = row["movieId"]
        true_rating = row["rating"]

        user_orig = raw_um.loc[user]
        user_norm = norm_um.loc[user]

        pred = predict_rating(user_orig, user_norm, movie, item_sim, k)

        y_true.append(true_rating)
        y_pred.append(pred)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    return mae, rmse


# ------------------------------------------------------------
# Plots
# ------------------------------------------------------------
def plot_rmse_vs_k(df):
    combos = df[["normalization", "similarity"]].drop_duplicates()

    plt.figure(figsize=(10, 6))
    for _, row in combos.iterrows():
        norm = row["normalization"]
        sim = row["similarity"]

        subset = df[(df["normalization"] == norm) & (df["similarity"] == sim)]
        subset = subset.sort_values("top_k")

        plt.plot(subset["top_k"], subset["RMSE"], marker="o", label=f"{norm} + {sim}")

    plt.title("ItemCF RMSE vs top_k")
    plt.xlabel("top_k (neighbors)")
    plt.ylabel("RMSE")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_best_configs(df, top_n=10):
    best = df.sort_values("RMSE").head(top_n)

    labels = [
        f"{r['normalization']}\n{r['similarity']}\nmin={int(r['min_ratings'])},k={int(r['top_k'])}"
        for _, r in best.iterrows()
    ]

    plt.figure(figsize=(10, 6))
    plt.bar(range(top_n), best["RMSE"])
    plt.xticks(range(top_n), labels, rotation=45, ha="right")
    plt.title(f"Top {top_n} ItemCF Configurations")
    plt.ylabel("RMSE")
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# MAIN: Grid Search
# ------------------------------------------------------------
if __name__ == "__main__":
    results = []

    for norm in ["raw", "mean_center", "zscore"]:
        for sim in ["cosine", "pearson"]:
            for min_r in [3, 5, 10]:
                for k in [10, 20, 40, 60]:

                    mae, rmse = evaluate_itemcf(norm, sim, min_r, k)

                    print(f"{norm} + {sim}, min={min_r}, k={k} → RMSE={rmse:.4f}")

                    results.append({
                        "normalization": norm,
                        "similarity": sim,
                        "min_ratings": min_r,
                        "top_k": k,
                        "MAE": mae,
                        "RMSE": rmse
                    })

    df = pd.DataFrame(results)
    df.to_csv("itemcf_results.csv", index=False)
    print("\nSaved → itemcf_results.csv")

    # Plot results
    plot_rmse_vs_k(df)
    plot_best_configs(df)
