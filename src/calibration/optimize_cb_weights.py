
import sys
import os
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add parent directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from recommender.data_loader import load_movies, load_train_valid_test_splits
from recommender.CB.content_based import ContentBasedModel

# Define paths relative to the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

def manual_mean_squared_error(y_true, y_pred):
    return np.mean((np.array(y_true) - np.array(y_pred))**2)

def manual_mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))

def evaluate_scenario(name, weights, train_df, test_df, movies):
    print(f"\n>>> Running Scenario: {name}")
    print(f"    Weights: {weights}")
    
    model = ContentBasedModel(movies, train_df, weights=weights)
    
    preds = []
    trues = []
    
    # Evaluate on a subset of test set for speed if needed, but we do full here
    # 5k samples is enough for significance
    test_subset = test_df.sample(n=min(len(test_df), 5000), random_state=42)
    
    for i, row in test_subset.iterrows():
        uid = row['userId']
        mid = row['movieId']
        actual = row['rating']
        
        pred = model.predict_rating(uid, mid, top_k=20)
        
        if not np.isnan(pred):
            pred = min(5.0, max(0.5, pred))
            preds.append(pred)
            trues.append(actual)
            
    rmse = math.sqrt(manual_mean_squared_error(trues, preds))
    mae = manual_mean_absolute_error(trues, preds)
    coverage = len(preds) / len(test_subset)
    
    print(f"    RMSE: {rmse:.4f} | MAE: {mae:.4f} | Cov: {coverage:.2%}")
    return {"name": name, "weights": str(weights), "RMSE": rmse, "MAE": mae, "Coverage": coverage}

if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print("[1] Loading Data...")
    train_df, valid_df, test_df = load_train_valid_test_splits()
    movies = load_movies()
    
    # DEFINE SCENARIOS
    scenarios = [
        {
            "name": "Baseline",
            "weights": {'genres': 2, 'director': 2, 'keywords': 2, 'actors': 1, 'year': 1, 'overview': 1, 'companies': 0, 'countries': 0},
            "hypothesis": ""
        },
        {
            "name": "Global Cinema (Countries, KeyLow)",
            "weights": {'genres': 2, 'director': 2, 'keywords': 1, 'actors': 1, 'year': 1, 'overview': 1, 'companies': 0, 'countries': 2},
            "hypothesis": "Cultural/ Language factors (Korean cinema, French films) can cluster similar tastes."
        },
               {
            "name": "Global Cinema (Countries, KeyHigh)",
            "weights": {'genres': 2, 'director': 2, 'keywords': 2, 'actors': 1, 'year': 1, 'overview': 1, 'companies': 0, 'countries': 2},
            "hypothesis": "with different weights, Cultural/ Language factors (Korean cinema, French films) can cluster similar tastes."
        },
        {
            "name": "Global Cinema (Countries, countLow)",
            "weights": {'genres': 2, 'director': 2, 'keywords': 2, 'actors': 1, 'year': 1, 'overview': 1, 'companies': 0, 'countries': 1},
            "hypothesis": "with different weights, Cultural/ Language factors (Korean cinema, French films) can cluster similar tastes."
        },
        
        {
            "name": "Cultural Purist (Region Heavy)",
            "weights": {'genres': 2, 'director': 1, 'keywords': 1, 'actors': 1, 'year': 1, 'overview': 1, 'companies': 0, 'countries': 5},
            "hypothesis": "Country/ Language barrier is the toughest filter. Users usually stay in regional clusters like 'Korean cinema' or 'American blockbuster'."
        },
        {
            "name": "Production Quality (Context)",
            "weights": {'genres': 1, 'director': 2, 'keywords': 1, 'actors': 1, 'year': 1, 'overview': 1, 'companies': 2, 'countries': 2},
            "hypothesis": "Filmin 'kalitesini' ve 'bütçesini' belirleyen üçlü: Yönetmen, Stüdyo, Ülke. Bu üçünün kombinasyonu 'Premium' hissini yakalar."
        },
        {
            "name": "Narrative Deep Dive (NLP Focus)",
            "weights": {'genres': 1, 'director': 1, 'keywords': 0, 'actors': 1, 'year': 1, 'overview': 4, 'companies': 0, 'countries': 0},
            "hypothesis": "Keywords are noisy. Professional 'Overview' (Summary) text, TF-IDF, provides the best content matching."
        },
        {
            "name": "Time Capsule (Era Specific)",
            "weights": {'genres': 2, 'director': 1, 'keywords': 1, 'actors': 1, 'year': 5, 'overview': 1, 'companies': 0, 'countries': 0},
            "hypothesis": "Nostalgia factor."
        },
         {
            "name": "Cast & Crew (People)",
            "weights": {'genres': 1, 'director': 3, 'keywords': 1, 'actors': 3, 'year': 1, 'overview': 1, 'companies': 0, 'countries': 0},
            "hypothesis": "Personnel factor. No matter the movie's theme, it's watched for the liked actors and director."
        }
    ]
    
    results = []
    
    for scen in scenarios:
        res = evaluate_scenario(scen["name"], scen["weights"], train_df, test_df, movies)
        res["hypothesis"] = scen["hypothesis"]
        results.append(res)
        
    # Print Comparison
    print("\n\n=== EXPERIMENT RESULTS ===")
    df_res = pd.DataFrame(results)
    df_res = df_res.sort_values("RMSE")
    print(df_res[["name", "RMSE", "MAE", "Coverage"]])
    
    # Save to file
    out_path = os.path.join(RESULTS_DIR, "cb_weight_optimization_results.csv")
    df_res.to_csv(out_path, index=False)
    print(f"\nSaved detailed results to {out_path}")
