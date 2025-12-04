"""
Item-Based CF hyperparameter experiment with proper train/test split
and bias-correct prediction.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity

from src.recommender.data_loader import load_ratings


# ------------------------------------------------------------
# Train / Test split — user-based
# ------------------------------------------------------------
def train_test_split_user(ratings: pd.DataFrame,
                          test_ratio: float = 0.2,
                          min_items: int = 5):
    """
    Splits ratings into train and test per user.
    Only users with at least `min_items` ratings are used.
    """
    train_parts = []
    test_parts = []

    for user, group in ratings.groupby("userId"):
        if len(group) < min_items:
            # too few ratings → keep all in train
            train_parts.append(group)
            continue

        group = group.sample(frac=1.0, random_state=42)  # shuffle
        split_idx = int(len(group) * (1 - test_ratio))
        train_parts.append(group.iloc[:split_idx])
        test_parts.append(group.iloc[split_idx:])

    train = pd.concat(train_parts).reset_index(drop=True)
    test = pd.concat(test_parts).reset_index(drop=True) if test_parts else pd.DataFrame(columns=ratings.columns)

    return train, test


# ------------------------------------------------------------
# Build user–movie matrix with NaNs for missing
# ------------------------------------------------------------
def build_user_movie_matrix_raw(ratings: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot ratings into user×movie matrix with NaN as missing.
    """
    um = ratings.pivot_table(
        index="userId",
        columns="movieId",
        values="rating",
        aggfunc="mean"
    )
    return um


# ------------------------------------------------------------
# Normalization methods
# ------------------------------------------------------------
def normalize_raw(mat: pd.DataFrame) -> pd.DataFrame:
    return mat.fillna(0)


def normalize_mean_center(mat: pd.DataFrame) -> pd.DataFrame:
    mean = mat.mean(axis=1, skipna=True)
    centered = mat.sub(mean, axis=0)
    return centered.fillna(0)


def normalize_zscore(mat: pd.DataFrame) -> pd.DataFrame:
    mean = mat.mean(axis=1, skipna=True)
    std = mat.std(axis=1, skipna=True).replace(0, 1)
    z = mat.sub(mean, axis=0).div(std, axis=0)
    return z.fillna(0)


normalizers = {
    "raw": normalize_raw,
    "mean_center": normalize_mean_center,
    "zscore": normalize_zscore,
}


# ------------------------------------------------------------
# Pearson similarity
# ------------------------------------------------------------
def pearson_sim(mat_T: pd.DataFrame) -> np.ndarray:
    """
    mat_T: movies × users (transpose of user–movie)
    returns: movie×movie similarity matrix values
    """
    return mat_T.T.corr().fillna(0).values


# ------------------------------------------------------------
# Correct ItemCF prediction with bias reconstruction
# ------------------------------------------------------------
def predict_rating(user_orig: pd.Series,
                   user_norm: pd.Series,
                   movie_id: int,
                   item_sim: pd.DataFrame,
                   k: int) -> float:
    """
    user_orig: raw ratings for this user (NaN = no rating)
    user_norm: normalized ratings for this user (0 when missing)
    """
    if movie_id not in item_sim.columns:
        # no similarity info for this movie
        rated_mask = user_orig.notna()
        if rated_mask.any():
            return user_orig[rated_mask].mean()
        return 3.0  # fallback global-ish baseline

    sims = item_sim[movie_id].drop(movie_id, errors="ignore")

    if sims.empty:
        rated_mask = user_orig.notna()
        if rated_mask.any():
            return user_orig[rated_mask].mean()
        return 3.0

    neighbors = sims.sort_values(ascending=False).head(k)

    # only neighbors the user has actually rated
    rated_neighbors = [m for m in neighbors.index if pd.notna(user_orig.get(m))]
    if len(rated_neighbors) == 0:
        rated_mask = user_orig.notna()
        if rated_mask.any():
            return user_orig[rated_mask].mean()
        return 3.0

    weights = neighbors.loc[rated_neighbors]
    ratings = user_norm.loc[rated_neighbors]

    if np.all(ratings.values == 0):
        rated_mask = user_orig.notna()
        if rated_mask.any():
            return user_orig[rated_mask].mean()
        return 3.0

    weighted_sum = np.dot(weights.values, ratings.values) / np.sum(np.abs(weights.values))

    # reconstruct to original rating scale
    rated_mask = user_orig.notna()
    if rated_mask.any():
        user_mean = user_orig[rated_mask].mean()
    else:
        user_mean = 3.0

    return float(user_mean + weighted_sum)


# ------------------------------------------------------------
# Evaluation for a single hyperparameter config
# ------------------------------------------------------------
def evaluate_itemcf(norm: str,
                    sim_type: str,
                    min_ratings: int,
                    k: int,
                    test_ratio: float = 0.2):
    """
    Returns (MAE, RMSE) for a given hyperparameter config.
    Uses user-based train/test split.
    """
    ratings = load_ratings()

    # filter movies by minimum rating count
    counts = ratings["movieId"].value_counts()
    valid_movies = counts[counts >= min_ratings].index
    ratings = ratings[ratings["movieId"].isin(valid_movies)]

    # train/test split per user
    train, test = train_test_split_user(ratings, test_ratio=test_ratio, min_items=5)
    if test.empty:
        raise ValueError("Test set is empty after splitting. Check data or parameters.")

    # build user–movie matrices from TRAIN only
    raw_um = build_user_movie_matrix_raw(train)
    norm_um = normalizers[norm](raw_um)

    # similarity matrix
    if sim_type == "cosine":
        sim_values = cosine_similarity(norm_um.T)
    else:
        sim_values = pearson_sim(norm_um.T)

    item_sim = pd.DataFrame(sim_values, index=norm_um.columns, columns=norm_um.columns)

    # evaluation on TEST set
    y_true, y_pred = [], []

    # optional: to speed up, sample subset of test
    test_sample = test.sample(min(3000, len(test)), random_state=42)

    for _, row in test_sample.iterrows():
        user = row["userId"]
        movie = row["movieId"]
        true_rating = row["rating"]

        if user not in raw_um.index:
            # user has no train data (edge case)
            continue
        if movie not in raw_um.columns:
            # movie not in similarity matrix
            continue

        user_orig = raw_um.loc[user]
        user_norm = norm_um.loc[user]

        pred = predict_rating(user_orig, user_norm, movie, item_sim, k)
        y_true.append(true_rating)
        y_pred.append(pred)

    if not y_true:
        raise ValueError("No valid (user, movie) pairs in test after filtering.")

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse


# ------------------------------------------------------------
# Plot helpers
# ------------------------------------------------------------
def plot_rmse_vs_k(df: pd.DataFrame):
    combos = df[["normalization", "similarity"]].drop_duplicates()

    plt.figure(figsize=(9, 5))
    for _, row in combos.iterrows():
        norm = row["normalization"]
        sim = row["similarity"]
        subset = df[(df["normalization"] == norm) & (df["similarity"] == sim)]
        subset = subset.sort_values("top_k")

        plt.plot(subset["top_k"], subset["RMSE"], marker="o", label=f"{norm} + {sim}")

    plt.xlabel("top_k (neighbors)")
    plt.ylabel("RMSE")
    plt.title("ItemCF: RMSE vs top_k")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_best_configs(df: pd.DataFrame, top_n: int = 10):
    best = df.sort_values("RMSE").head(top_n)

    labels = [
        f"{r['normalization']}\n{r['similarity']}\nmin={int(r['min_ratings'])},k={int(r['top_k'])}"
        for _, r in best.iterrows()
    ]

    plt.figure(figsize=(9, 5))
    plt.bar(range(len(best)), best["RMSE"])
    plt.xticks(range(len(best)), labels, rotation=45, ha="right")
    plt.ylabel("RMSE")
    plt.title(f"Top {top_n} ItemCF configs (lower is better)")
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
if __name__ == "__main__":
    results = []

    for norm in ["raw", "mean_center", "zscore"]:
        for sim in ["cosine", "pearson"]:
            for min_r in [3, 5, 10]:
                for k in [10, 20, 40, 60]:
                    mae, rmse = evaluate_itemcf(norm, sim, min_r, k, test_ratio=0.2)
                    print(f"{norm} + {sim}, min={min_r}, k={k} → MAE={mae:.4f}  RMSE={rmse:.4f}")

                    results.append({
                        "normalization": norm,
                        "similarity": sim,
                        "min_ratings": min_r,
                        "top_k": k,
                        "MAE": mae,
                        "RMSE": rmse,
                    })

    df = pd.DataFrame(results)
    df.to_csv("itemcf_results.csv", index=False)
    print("\nSaved → itemcf_results.csv")

    # plots
    plot_rmse_vs_k(df)
    plot_best_configs(df, top_n=10)
