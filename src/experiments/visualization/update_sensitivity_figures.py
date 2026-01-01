"""
Update sensitivity analysis figures using current IBCF results
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Read the IBCF results
results_dir = Path(__file__).parent / "results" / "item_based_cf"
df = pd.read_csv(results_dir / "itemcf_results.csv")

# Filter for cosine similarity only
df_filtered = df[df['similarity'] == 'cosine'].copy()

# ============================================================================
# Figure 2: Sensitivity to K (Top K Neighbors)
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

# Get unique values
k_values = sorted(df_filtered['top_k'].unique())

# Calculate MAXIMUM NDCG for each configuration to find top performers
config_performance = []
for norm in df_filtered['normalization'].unique():
    for min_r in df_filtered['min_ratings'].unique():
        subset = df_filtered[(df_filtered['normalization'] == norm) & 
                            (df_filtered['min_ratings'] == min_r)]
        max_ndcg = subset['NDCG'].max()  # MAX instead of mean
        max_k = subset.loc[subset['NDCG'].idxmax(), 'top_k']  # K value where max occurs
        config_performance.append({
            'normalization': norm,
            'min_ratings': min_r,
            'max_ndcg': max_ndcg,
            'best_k': max_k
        })

# Sort by MAXIMUM NDCG and take top 5
config_df = pd.DataFrame(config_performance)
config_df = config_df.sort_values('max_ndcg', ascending=False).head(5)

# Define distinct colors for top 5
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
markers = ['o', 's', '^', 'D', 'v']

# Plot only top 5 configurations
for idx, (_, config) in enumerate(config_df.iterrows()):
    norm_method = config['normalization']
    min_rating = config['min_ratings']
    
    subset = df_filtered[(df_filtered['normalization'] == norm_method) & 
                        (df_filtered['min_ratings'] == min_rating)]
    means = []
    
    for k in k_values:
        k_data = subset[subset['top_k'] == k]['NDCG']
        if len(k_data) > 0:
            means.append(k_data.mean())
        else:
            means.append(np.nan)
    
    means = np.array(means)
    
    # Create clean label with MAX NDCG and best K
    label = f'{norm_method}, min_ratings={int(min_rating)} (max: {config["max_ndcg"]:.4f} @K={int(config["best_k"])})'
    ax.plot(k_values, means, 
            marker=markers[idx],
            color=colors[idx],
            label=label, linewidth=2.5, markersize=8, alpha=0.85)

ax.set_xlabel('Top K Neighbors', fontsize=12, fontweight='bold', color='white')
ax.set_ylabel('NDCG@10', fontsize=12, fontweight='bold', color='white')
ax.set_title('Sensitivity Analysis: Impact of K on NDCG@10\n(Top 5 by Maximum NDCG)', 
             fontsize=14, fontweight='bold', color='white')
ax.legend(title='Configuration (sorted by max NDCG)', fontsize=9, loc='best', 
          facecolor='#1a1a1a', edgecolor='white', framealpha=0.9, 
          labelcolor='white', title_fontproperties={'weight': 'bold', 'size': 9})
ax.grid(True, alpha=0.3, linestyle='--', color='white')
ax.set_facecolor('#151515')
fig.patch.set_facecolor('#151515')
ax.tick_params(axis='both', colors='white', labelsize=10)
ax.spines['bottom'].set_color('white')
ax.spines['top'].set_color('white')
ax.spines['left'].set_color('white')
ax.spines['right'].set_color('white')

plt.tight_layout()
plt.savefig(results_dir / "fig_2_sensitivity_k.svg", format='svg', dpi=300, facecolor='#151515')
print("✓ Figure 2 saved: fig_2_sensitivity_k.svg (dark theme)")
plt.close()

# ============================================================================
# Figure 3: Sensitivity to Min Ratings Filter
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

min_ratings_values = sorted(df_filtered['min_ratings'].unique())

# Calculate MAXIMUM NDCG for each K configuration to find top performers
k_config_performance = []
for norm in df_filtered['normalization'].unique():
    for k in k_values:
        subset = df_filtered[(df_filtered['normalization'] == norm) & 
                            (df_filtered['top_k'] == k)]
        max_ndcg = subset['NDCG'].max()  # MAX instead of mean
        best_min_r = subset.loc[subset['NDCG'].idxmax(), 'min_ratings']  # min_ratings where max occurs
        k_config_performance.append({
            'normalization': norm,
            'top_k': k,
            'max_ndcg': max_ndcg,
            'best_min_ratings': best_min_r
        })

# Sort by MAXIMUM NDCG and take top 5
k_config_df = pd.DataFrame(k_config_performance)
k_config_df = k_config_df.sort_values('max_ndcg', ascending=False).head(5)

# Plot only top 5 K configurations
for idx, (_, config) in enumerate(k_config_df.iterrows()):
    norm_method = config['normalization']
    k = config['top_k']
    
    subset = df_filtered[(df_filtered['normalization'] == norm_method) & 
                        (df_filtered['top_k'] == k)]
    means = []
    
    for min_rating in min_ratings_values:
        mr_data = subset[subset['min_ratings'] == min_rating]['NDCG']
        if len(mr_data) > 0:
            means.append(mr_data.mean())
        else:
            means.append(np.nan)
    
    means = np.array(means)
    
    # Create clean label with MAX NDCG and best min_ratings
    label = f'{norm_method}, K={int(k)} (max: {config["max_ndcg"]:.4f} @min_r={int(config["best_min_ratings"])})'
    ax.plot(min_ratings_values, means,
            marker=markers[idx],
            color=colors[idx],
            label=label, linewidth=2.5, markersize=8, alpha=0.85)

ax.set_xlabel('Min Ratings Filter', fontsize=12, fontweight='bold', color='white')
ax.set_ylabel('NDCG@10', fontsize=12, fontweight='bold', color='white')
ax.set_title('Sensitivity Analysis: Impact of Min Ratings on NDCG@10\n(Top 5 by Maximum NDCG)', 
             fontsize=14, fontweight='bold', color='white')
ax.legend(title='Configuration (sorted by max NDCG)', fontsize=9, loc='best',
          facecolor='#1a1a1a', edgecolor='white', framealpha=0.9,
          labelcolor='white', title_fontproperties={'weight': 'bold', 'size': 9})
ax.grid(True, alpha=0.3, linestyle='--', color='white')
ax.set_facecolor('#151515')
fig.patch.set_facecolor('#151515')
ax.tick_params(axis='both', colors='white', labelsize=10)
ax.spines['bottom'].set_color('white')
ax.spines['top'].set_color('white')
ax.spines['left'].set_color('white')
ax.spines['right'].set_color('white')

plt.tight_layout()
plt.savefig(results_dir / "fig_3_sensitivity_min_ratings.svg", format='svg', dpi=300, facecolor='#151515')
print("✓ Figure 3 saved: fig_3_sensitivity_min_ratings.svg (dark theme)")
plt.close()

print("\n✅ Both figures updated successfully!")
print(f"   - Using data from: itemcf_results.csv")
print(f"   - Showing: Top 5 configurations by MAXIMUM NDCG (not average)")
print(f"   - Legend shows: max NDCG value + where it occurs")
print(f"   - Similarity: cosine")
print(f"   - Background: #151515 (dark theme)")
