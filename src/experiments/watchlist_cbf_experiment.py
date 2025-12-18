"""
Watchlist tabanlı içerik önerisi için basit bir LOO (leave-one-out) değerlendirme.

Ölçütler:
- Hit@5, Hit@10
- MRR

Grid:
- TF-IDF ngram, max_features, min_df, başlık ağırlığı gibi parametreler denenir.
"""

import os
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from recommender.CBF.content_based import ContentBasedModel
from recommender.watchlist.movie_encoder import build_movie_vectors
from recommender.watchlist.similarity import cosine_sim

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
MOVIES_PATH = os.path.join(DATA_DIR, "movies_tmdb.csv")
WATCHLIST_PATH = os.path.join(DATA_DIR, "watchlist.csv")
RATINGS_PATH = os.path.join(DATA_DIR, "ratings.csv")


def load_data():
    movies = pd.read_csv(MOVIES_PATH)
    watchlist = pd.read_csv(WATCHLIST_PATH)
    watchlist = watchlist.dropna(subset=["userId", "movieId"])
    watchlist["userId"] = watchlist["userId"].astype(int)
    watchlist["movieId"] = watchlist["movieId"].astype(int)
    return movies, watchlist


def load_data_with_ratings():
    movies = pd.read_csv(MOVIES_PATH)
    watchlist = pd.read_csv(WATCHLIST_PATH)
    ratings = pd.read_csv(RATINGS_PATH)

    watchlist = watchlist.dropna(subset=["userId", "movieId"])
    watchlist["userId"] = watchlist["userId"].astype(int)
    watchlist["movieId"] = watchlist["movieId"].astype(int)

    ratings = ratings.dropna(subset=["userId", "movieId", "rating"])
    ratings["userId"] = ratings["userId"].astype(int)
    ratings["movieId"] = ratings["movieId"].astype(int)
    ratings["rating"] = pd.to_numeric(ratings["rating"], errors="coerce")
    ratings = ratings.dropna(subset=["rating"])

    return movies, watchlist, ratings


def leave_one_out_eval(
    movies_df: pd.DataFrame,
    watchlist_df: pd.DataFrame,
    movie_vectors: Dict[int, np.ndarray],
    k_list: List[int] = [5, 10],
) -> Dict[str, float]:
    all_movie_ids = set(movies_df["movieId"].astype(int))

    eval_count = 0
    hit_at = {k: 0 for k in k_list}
    mrr_sum = 0.0

    for uid, group in watchlist_df.groupby("userId"):
        user_movies = group["movieId"].astype(int).unique().tolist()
        if len(user_movies) < 2:
            continue  # LOO yapacak kadar film yok

        for held_out in user_movies:
            train_movies = [m for m in user_movies if m != held_out]

            train_vecs = [movie_vectors[m] for m in train_movies if m in movie_vectors]
            held_vec = movie_vectors.get(held_out)
            if not train_vecs or held_vec is None:
                continue

            profile = np.mean(train_vecs, axis=0)

            candidates = all_movie_ids - set(train_movies)
            scored = []
            for mid in candidates:
                vec = movie_vectors.get(mid)
                if vec is None:
                    continue
                scored.append((mid, cosine_sim(profile, vec)))

            if not scored:
                continue

            ranked = sorted(scored, key=lambda x: x[1], reverse=True)
            eval_count += 1

            # rank bul
            positions = [i for i, (mid, _) in enumerate(ranked) if mid == held_out]
            if not positions:
                continue
            pos = positions[0] + 1  # 1-based

            for k in k_list:
                if pos <= k:
                    hit_at[k] += 1
            mrr_sum += 1.0 / pos

    if eval_count == 0:
        return {f"hit@{k}": 0.0 for k in k_list} | {"mrr": 0.0, "evals": 0}

    metrics = {
        f"hit@{k}": hit_at[k] / eval_count for k in k_list
    }
    metrics["mrr"] = mrr_sum / eval_count
    metrics["evals"] = eval_count
    return metrics


def _build_profile_from_watchlist(
    train_movie_ids: Sequence[int],
    movie_to_idx: Dict[int, int],
    tfidf_matrix,
) -> Optional[np.ndarray]:
    indices = [movie_to_idx[mid] for mid in train_movie_ids if mid in movie_to_idx]
    if not indices:
        return None

    # Ortalama içerik vektörü (1 x d)
    profile = tfidf_matrix[indices].mean(axis=0)
    return np.asarray(profile).reshape(1, -1)


def leave_one_out_eval_content_based(
    movies_df: pd.DataFrame,
    watchlist_df: pd.DataFrame,
    model: ContentBasedModel,
    k_list: List[int] = [5, 10],
) -> Dict[str, float]:
    """
    ContentBasedModel'in TF-IDF temsilini kullanarak watchlist LOO değerlendirmesi.
    Kullanıcı profili, watchlist'teki filmlerin ortalaması olarak hesaplanır.
    """
    all_movie_ids = set(movies_df["movieId"].astype(int))
    eval_count = 0
    hit_at = {k: 0 for k in k_list}
    mrr_sum = 0.0

    tfidf_matrix = model.tfidf_matrix
    movie_to_idx = model.movie_to_idx

    for uid, group in watchlist_df.groupby("userId"):
        user_movies = group["movieId"].astype(int).unique().tolist()
        if len(user_movies) < 2:
            continue

        for held_out in user_movies:
            train_movies = [m for m in user_movies if m != held_out]
            profile = _build_profile_from_watchlist(train_movies, movie_to_idx, tfidf_matrix)
            if profile is None:
                continue

            candidates = all_movie_ids - set(train_movies)
            scored = []
            for mid in candidates:
                idx = movie_to_idx.get(mid)
                if idx is None:
                    continue
                movie_vec = tfidf_matrix[idx]
                score = cosine_similarity(profile, movie_vec)[0][0]
                scored.append((mid, score))

            if not scored:
                continue

            ranked = sorted(scored, key=lambda x: x[1], reverse=True)
            eval_count += 1

            positions = [i for i, (mid, _) in enumerate(ranked) if mid == held_out]
            if not positions:
                continue
            pos = positions[0] + 1

            for k in k_list:
                if pos <= k:
                    hit_at[k] += 1
            mrr_sum += 1.0 / pos

    if eval_count == 0:
        return {f"hit@{k}": 0.0 for k in k_list} | {"mrr": 0.0, "evals": 0}

    metrics = {f"hit@{k}": hit_at[k] / eval_count for k in k_list}
    metrics["mrr"] = mrr_sum / eval_count
    metrics["evals"] = eval_count
    return metrics


def run_grid():
    movies_df, watchlist_df = load_data()

    configs = [
        {
            "name": "tfidf_unigram_5k",
            "params": dict(max_features=5000, ngram_range=(1, 1), min_df=1, stop_words="english"),
        },
        {
            "name": "tfidf_bigram_10k",
            "params": dict(max_features=10000, ngram_range=(1, 2), min_df=2, stop_words="english"),
        },
        {
            "name": "tfidf_bigram_title_boost",
            "params": dict(
                max_features=15000,
                ngram_range=(1, 2),
                min_df=2,
                stop_words="english",
                field_weights={"title": 1.0, "genres": 1.2, "keywords": 1.0, "overview": 0.8},
            ),
        },
    ]

    results = []
    for cfg in configs:
        print(f"\n=== Config: {cfg['name']} ===")
        movie_vectors = build_movie_vectors(movies_df, **cfg["params"])
        metrics = leave_one_out_eval(movies_df, watchlist_df, movie_vectors)
        row = {
            "config": cfg["name"],
            **{f"param_{k}": v for k, v in cfg["params"].items()},
            **metrics,
        }
        results.append(row)
        print(f"evals={metrics['evals']}, hit@5={metrics['hit@5']:.4f}, hit@10={metrics['hit@10']:.4f}, mrr={metrics['mrr']:.4f}")

    # ContentBasedModel (soup) denemesi
    print("\n=== Config: content_based_soup ===")
    movies_df_cb, watchlist_df_cb, ratings_df = load_data_with_ratings()
    cb_model = ContentBasedModel(movies_df_cb, ratings_df)
    metrics_cb = leave_one_out_eval_content_based(movies_df_cb, watchlist_df_cb, cb_model)
    row_cb = {
        "config": "content_based_soup",
        "param_model": "ContentBasedModel",
        **metrics_cb,
    }
    results.append(row_cb)
    print(f"evals={metrics_cb['evals']}, hit@5={metrics_cb['hit@5']:.4f}, hit@10={metrics_cb['hit@10']:.4f}, mrr={metrics_cb['mrr']:.4f}")

    df = pd.DataFrame(results)
    out_path = os.path.join(os.path.dirname(__file__), "watchlist_cbf_results.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSonuçlar kaydedildi: {out_path}")
    print(df[["config", "hit@5", "hit@10", "mrr", "evals"]])


if __name__ == "__main__":
    run_grid()

