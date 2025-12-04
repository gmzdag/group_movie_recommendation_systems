"""
Evaluation script for UBCF with:
- significance weighting
- shrinkage pearson
"""

import sys
import os
# Add parent directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

from recommender.data_loader import load_all_data
from recommender.UBCF.similarity_user import pearson_sw, pearson_shrink
from recommender.UBCF.neighbors_user import load_or_compute_neighbors
from recommender.UBCF.user_based_cf import UserBasedCF


CACHE_SW = "../../cache/user_neighbors_sw.pkl"
CACHE_SHR = "../../cache/user_neighbors_shrink.pkl"


print("[1] Loading data...")
movies, ratings, watchlists, R_cf, R_dense = load_all_data()

ratings_clean = ratings.groupby(["userId","movieId"])["rating"].mean().reset_index()

print("[2] Train/test split...")
train_df, test_df = train_test_split(ratings_clean, test_size=0.2, random_state=42)

R_train = train_df.pivot(index="userId", columns="movieId", values="rating")
R_train = R_train.reindex(index=R_cf.index, columns=R_cf.columns)

global_mean = train_df["rating"].mean()
user_means = R_train.mean(axis=1)
item_means = R_train.mean(axis=0)

print("[3] Loading neighbors...")
neighbors_sw = load_or_compute_neighbors(CACHE_SW, R_train, pearson_sw)
neighbors_shr = load_or_compute_neighbors(CACHE_SHR, R_train, pearson_shrink)

model_sw  = UserBasedCF(R_train, neighbors_sw,  user_means, item_means, global_mean)
model_shr = UserBasedCF(R_train, neighbors_shr, user_means, item_means, global_mean)


# ----------- Evaluation -----------
def evaluate(model):
    preds, trues = [], []
    for _, row in test_df.iterrows():
        u, m, true_r = row["userId"], row["movieId"], row["rating"]

        if m not in R_train.columns:
            continue

        preds.append(model.predict(u, m))
        trues.append(true_r)

    rmse = math.sqrt(mean_squared_error(trues, preds))
    mae = mean_absolute_error(trues, preds)
    coverage = len(preds) / len(test_df)

    return rmse, mae, coverage, preds, trues


print("[4] Testing models...")

rmse_sw, mae_sw, cov_sw, preds_sw, trues_sw = evaluate(model_sw)
rmse_shr, mae_shr, cov_shr, preds_shr, trues_shr = evaluate(model_shr)


print("\n=== SIGNIFICANCE WEIGHTING ===")
print("RMSE:", rmse_sw)
print("MAE :", mae_sw)
print("Coverage:", cov_sw)

print("\n=== SHRINKAGE PEARSON ===")
print("RMSE:", rmse_shr)
print("MAE :", mae_shr)
print("Coverage:", cov_shr)


# ---------- Plots ----------
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.hist(np.array(preds_sw) - np.array(trues_sw), bins=30, alpha=0.7)
plt.title("Error Dist - Significance Weighting")

plt.subplot(1,2,2)
plt.hist(np.array(preds_shr) - np.array(trues_shr), bins=30, alpha=0.7, color='orange')
plt.title("Error Dist - Shrinkage")
plt.show()
