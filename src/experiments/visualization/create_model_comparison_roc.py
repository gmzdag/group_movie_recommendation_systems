"""
Model Comparison ROC Curves - ALL MODELS
Creates ROC curves comparing UBCF, IBCF, and CBF models
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, auc
from sklearn.metrics.pairwise import cosine_similarity
import sys
import os
from functools import partial

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from recommender.data_loader import load_all_data, build_cf_matrix
from recommender.UBCF.similarity_user import cosine_sim
from recommender.UBCF.neighbors_user import load_or_compute_neighbors
from recommender.UBCF.user_based_cf import UserBasedCF
from recommender.CB.content_based import ContentBasedModel

# Load data
print("\n" + "="*70)
print("LOADING DATA")
print("="*70)

PROJECT_ROOT = Path(__file__).parent.parent.parent
data_dir = PROJECT_ROOT / "data"
splits_dir = data_dir / "splits"

# Load train/test splits
train_df = pd.read_csv(splits_dir / "train.csv")
test_df = pd.read_csv(splits_dir / "test.csv")
movies_df = pd.read_csv(data_dir / "movies_TMDB.csv")

print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

# Build rating matrices
R_train = train_df.pivot(index="userId", columns="movieId", values="rating")
user_means = R_train.mean(axis=1)
item_means = R_train.mean(axis=1)
global_mean = train_df["rating"].mean()

# Rating threshold
RATING_THRESHOLD = 3.5

# Prepare test data
test_df['is_relevant'] = (test_df['rating'] >= RATING_THRESHOLD).astype(int)

# Sample 2000 test samples for more reliable results
if len(test_df) > 2000:
    test_sample = test_df.sample(n=2000, random_state=42)
else:
    test_sample = test_df.copy()

print(f"Using {len(test_sample)} test samples")
print(f"Positive: {test_sample['is_relevant'].sum()}, Negative: {(~test_sample['is_relevant'].astype(bool)).sum()}")

# Initialize models
print("\n" + "="*70)
print("INITIALIZING MODELS")
print("="*70)

models = {}

# 1. UBCF
print("\n1️⃣  Initializing UBCF...")
sim_func = partial(cosine_sim, MIN_OVERLAP=5)
neighbors = load_or_compute_neighbors(R_train, sim_func, K=20, metric='Cosine_Overlap5')
ubcf = UserBasedCF(R=R_train, neighbors=neighbors, user_means=user_means, 
                   item_means=item_means, global_mean=global_mean)
models['UBCF'] = ubcf
print("   ✓ UBCF initialized")

# 2. IBCF - Build from scratch
print("\n2️⃣  Initializing IBCF...")
# Build normalized matrix
raw_um = build_cf_matrix(train_df)
norm_um = raw_um.sub(raw_um.mean(axis=1), axis=0)  # Mean center
norm_um_filled = norm_um.fillna(0)

# Compute item similarity
item_sim_matrix = cosine_similarity(norm_um_filled.T)
item_sim = pd.DataFrame(item_sim_matrix, index=raw_um.columns, columns=raw_um.columns)
np.fill_diagonal(item_sim.values, 0)

# IBCF prediction function
def ibcf_predict(user_id, movie_id, raw_um, norm_um, item_sim, user_means, global_mean, top_k=50):
    if user_id not in raw_um.index or movie_id not in item_sim.index:
        return global_mean
    
    user_ratings_norm = norm_um.loc[user_id]
    rated_items = raw_um.loc[user_id].dropna().index
    valid_items = [item for item in rated_items if item != movie_id and item in item_sim.index]
    
    if len(valid_items) == 0:
        return global_mean
    
    sims = item_sim[movie_id].loc[valid_items]
    top_k_indices = sims.nlargest(top_k).index
    top_k_sims = sims.loc[top_k_indices]
    top_k_ratings = user_ratings_norm.loc[top_k_indices]
    
    if top_k_sims.abs().sum() == 0:
        return global_mean
    
    numerator = np.dot(top_k_sims.values, top_k_ratings.values)
    denominator = np.sum(np.abs(top_k_sims.values))
    
    if denominator == 0:
        return global_mean
    
    pred_norm = numerator / denominator
    pred = user_means[user_id] + pred_norm
    return max(0, min(5, pred))

models['IBCF'] = {
    'predict': lambda uid, mid: ibcf_predict(uid, mid, raw_um, norm_um, item_sim, user_means, global_mean)
}
print("   ✓ IBCF initialized")

# 3. CBF
print("\n3️⃣  Initializing CBF...")
cbf = ContentBasedModel(movies_df=movies_df, ratings_df=train_df)
models['CBF'] = cbf
print("   ✓ CBF initialized")

# 4. Hybrid1 (IBCF + CBF with Trust)
print("\n4️⃣  Initializing Hybrid1...")
from recommender.IBCF.neighbors_item import load_or_compute_item_neighbors

# Compute item neighbors for Hybrid1
item_neighbors = load_or_compute_item_neighbors(item_sim, K=60, metric="cosine_zscore")

# Hybrid1 prediction function
def hybrid1_predict(user_id, movie_id, raw_um, norm_um, item_neighbors, cbf, user_means, global_mean, C=1.0, top_k=60):
    # IBCF part
    ibcf_pred = np.nan
    n_neighbors = 0
    
    if user_id in raw_um.index and movie_id in item_neighbors:
        user_ratings_norm = norm_um.loc[user_id]
        rated_items = raw_um.loc[user_id].dropna().index
        
        neighbors_list = item_neighbors.get(movie_id, [])
        valid_neighbors = [nid for nid in neighbors_list if nid in rated_items and nid != movie_id]
        
        if len(valid_neighbors) > 0:
            n_neighbors = min(len(valid_neighbors), top_k)
            top_neighbors = valid_neighbors[:n_neighbors]
            
            sims = item_sim[movie_id].loc[top_neighbors]
            ratings = user_ratings_norm.loc[top_neighbors]
            
            if sims.abs().sum() > 0:
                numerator = np.dot(sims.values, ratings.values)
                denominator = np.sum(np.abs(sims.values))
                pred_norm = numerator / denominator
                ibcf_pred = user_means[user_id] + pred_norm
    
    # CBF part
    cbf_pred = cbf.predict_rating(user_id, movie_id)
    
    # Hybrid combination with trust
    if np.isnan(ibcf_pred) and np.isnan(cbf_pred):
        return global_mean
    elif np.isnan(ibcf_pred):
        return max(0, min(5, cbf_pred))
    elif np.isnan(cbf_pred):
        return max(0, min(5, ibcf_pred))
    
    alpha = n_neighbors / (n_neighbors + C)
    beta = C / (n_neighbors + C)
    final = alpha * ibcf_pred + beta * cbf_pred
    return max(0, min(5, final))

models['Hybrid1'] = {
    'predict': lambda uid, mid: hybrid1_predict(uid, mid, raw_um, norm_um, item_neighbors, cbf, user_means, global_mean)
}
print("   ✓ Hybrid1 initialized")

# 5. Hybrid2 (UBCF + CBF weighted)
print("\n5️⃣  Initializing Hybrid2...")

def hybrid2_predict(user_id, movie_id, ubcf, cbf, global_mean, w_ubcf=0.10, w_cbf=0.90):
    try:
        ubcf_pred = ubcf.predict(user_id, movie_id)
    except:
        ubcf_pred = np.nan
    
    cbf_pred = cbf.predict_rating(user_id, movie_id)
    
    if np.isnan(ubcf_pred) and np.isnan(cbf_pred):
        return global_mean
    elif np.isnan(ubcf_pred):
        return max(0, min(5, cbf_pred))
    elif np.isnan(cbf_pred):
        return max(0, min(5, ubcf_pred))
    
    final = w_ubcf * ubcf_pred + w_cbf * cbf_pred
    return max(0, min(5, final))

models['Hybrid2'] = {
    'predict': lambda uid, mid: hybrid2_predict(uid, mid, ubcf, cbf, global_mean)
}
print("   ✓ Hybrid2 initialized")

# 6. Watchlist-based (CBF + Watchlist Boost)
print("\n6️⃣  Initializing Watchlist...")

# Load watchlist data
try:
    watchlist_df = pd.read_csv(data_dir / "watchlist.csv")
    user_watchlists = watchlist_df.groupby('userId')['movieId'].apply(set).to_dict()
    
    def watchlist_predict(user_id, movie_id, cbf, user_watchlists, global_mean):
        # Get CBF base score
        cbf_score = cbf.predict_rating(user_id, movie_id)
        
        # Check if movie is in user's watchlist
        if user_id in user_watchlists and movie_id in user_watchlists[user_id]:
            # Boost score for watchlist items
            if np.isnan(cbf_score):
                return 4.5  # High score for watchlist items
            else:
                return min(5.0, cbf_score * 1.2)  # 20% boost
        else:
            # Regular CBF score
            if np.isnan(cbf_score):
                return global_mean
            return max(0, min(5, cbf_score))
    
    models['Watchlist'] = {
        'predict': lambda uid, mid: watchlist_predict(uid, mid, cbf, user_watchlists, global_mean)
    }
    print("   ✓ Watchlist initialized")
    has_watchlist = True
except:
    print("   ⚠️  Watchlist data not found, skipping")
    has_watchlist = False

# Generate predictions
print("\n" + "="*70)
print("GENERATING PREDICTIONS")
print("="*70)

def get_predictions(model, test_data, model_name):
    print(f"\n📊 {model_name}...")
    
    y_true = []
    y_scores = []
    
    for idx, row in test_data.iterrows():
        user_id = row['userId']
        movie_id = row['movieId']
        true_label = row['is_relevant']
        
        try:
            if model_name == "UBCF":
                pred_score = model.predict(user_id, movie_id)
            elif model_name in ["IBCF", "Hybrid1", "Hybrid2", "Watchlist"]:
                pred_score = model['predict'](user_id, movie_id)
            elif model_name == "CBF":
                pred_score = model.predict_rating(user_id, movie_id)
            
            if pd.isna(pred_score) or pred_score is None:
                continue
            
            pred_score = max(0, min(5, pred_score))
            y_true.append(true_label)
            y_scores.append(pred_score)
            
        except:
            continue
    
    print(f"   ✓ {len(y_scores)}/{len(test_data)} predictions")
    return np.array(y_true), np.array(y_scores)

roc_data = {}

for model_name, model in models.items():
    y_true, y_scores = get_predictions(model, test_sample, model_name)
    
    if len(y_true) > 10:
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        roc_data[model_name] = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc}
        print(f"   ✓ AUC: {roc_auc:.4f}")

# Colors
model_colors = {
    'UBCF': '#3498db',
    'IBCF': '#e74c3c',
    'CBF': '#2ecc71',
    'Hybrid1': '#f39c12',
    'Hybrid2': '#9b59b6',
    'Watchlist': '#1abc9c'
}
model_linestyles = {
    'UBCF': '-',
    'IBCF': '-',
    'CBF': '-',
    'Hybrid1': '--',
    'Hybrid2': '--',
    'Watchlist': ':'
}

def create_roc_plot(theme='dark'):
    if theme == 'dark':
        bg_color, text_color, grid_color, legend_bg, suffix = '#151515', 'white', 'white', '#1a1a1a', ''
    else:
        bg_color, text_color, grid_color, legend_bg, suffix = 'white', 'black', 'gray', 'white', '_white'
    
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    
    # Plot all available models
    model_order = ['UBCF', 'IBCF', 'CBF', 'Hybrid1', 'Hybrid2', 'Watchlist']
    for model_name in model_order:
        if model_name in roc_data:
            data = roc_data[model_name]
            ax.plot(data['fpr'], data['tpr'], color=model_colors[model_name],
                   linestyle=model_linestyles[model_name], linewidth=2.5,
                   label=f"{model_name} (AUC = {data['auc']:.4f})", alpha=0.9)
    
    ax.plot([0, 1], [0, 1], color=grid_color, linestyle=':', linewidth=2,
           label='Random (AUC = 0.5000)', alpha=0.5)
    
    ax.set_xlabel('False Positive Rate', fontsize=13, fontweight='bold', color=text_color)
    ax.set_ylabel('True Positive Rate', fontsize=13, fontweight='bold', color=text_color)
    ax.set_title('ROC Curves: Model Comparison\n(Rating Threshold ≥ 3.5)', 
                 fontsize=15, fontweight='bold', color=text_color, pad=20)
    ax.legend(loc='lower right', fontsize=11, facecolor=legend_bg,
             edgecolor=text_color, framealpha=0.95, labelcolor=text_color)
    ax.grid(True, alpha=0.3, linestyle='--', color=grid_color)
    ax.tick_params(axis='both', colors=text_color, labelsize=11)
    
    for spine in ax.spines.values():
        spine.set_color(text_color)
    
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    results_dir = Path(__file__).parent / "results" / "comparison"
    plt.savefig(results_dir / f'model_comparison_roc{suffix}.svg', 
               format='svg', dpi=300, facecolor=bg_color)
    print(f"\n✓ {theme} theme: model_comparison_roc{suffix}.svg")
    plt.close()

# Create both themes
print("\n" + "="*70)
print("CREATING ROC VISUALIZATIONS")
print("="*70)

create_roc_plot(theme='dark')
create_roc_plot(theme='white')

# Summary
print("\n" + "="*70)
print("✅ SUCCESS!")
print("="*70)
print("\n📊 Files:")
print("   - model_comparison_roc.svg (Dark #151515)")
print("   - model_comparison_roc_white.svg (White)")
print("\n📈 AUC Scores:")
for rank, (name, data) in enumerate(sorted(roc_data.items(), key=lambda x: x[1]['auc'], reverse=True), 1):
    print(f"   {rank}. {name:12s} - AUC: {data['auc']:.4f}")
print("\n" + "="*70)
