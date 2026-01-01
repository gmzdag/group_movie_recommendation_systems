import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the data
df = pd.read_csv('results/itemcf_results.csv')

# Set professional style
plt.style.use('seaborn-v0_8-whitegrid')
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#BC4749']

# Create comprehensive comparison figure
fig, ax = plt.subplots(figsize=(14, 8))

# Define configurations to compare
configs = [
    {'norm': 'zscore', 'sim': 'cosine', 'min_r': 10, 'label': 'Z-Score + Cosine (min_r=10)', 'color': colors[0], 'marker': 'o', 'linewidth': 3},
    {'norm': 'zscore', 'sim': 'cosine', 'min_r': 5, 'label': 'Z-Score + Cosine (min_r=5)', 'color': colors[1], 'marker': 's', 'linewidth': 2.5},
    {'norm': 'mean_center', 'sim': 'cosine', 'min_r': 10, 'label': 'Mean-Center + Cosine (min_r=10)', 'color': colors[2], 'marker': '^', 'linewidth': 2.5},
    {'norm': 'mean_center', 'sim': 'cosine', 'min_r': 5, 'label': 'Mean-Center + Cosine (min_r=5)', 'color': colors[3], 'marker': 'D', 'linewidth': 2},
    {'norm': 'mean_center', 'sim': 'pearson', 'min_r': 10, 'label': 'Mean-Center + Pearson (min_r=10)', 'color': colors[4], 'marker': 'v', 'linewidth': 2},
    {'norm': 'mean_center', 'sim': 'pearson', 'min_r': 5, 'label': 'Mean-Center + Pearson (min_r=5)', 'color': colors[5], 'marker': 'p', 'linewidth': 1.5},
]

k_values = [10, 20, 40, 60]

# Plot each configuration
for config in configs:
    data = df[
        (df['normalization'] == config['norm']) & 
        (df['similarity'] == config['sim']) & 
        (df['min_ratings'] == config['min_r'])
    ]
    data_sorted = data.sort_values('top_k')
    
    ax.plot(data_sorted['top_k'], data_sorted['NDCG'], 
            marker=config['marker'], 
            linewidth=config['linewidth'], 
            markersize=10,
            label=config['label'],
            color=config['color'],
            alpha=0.85)

# Styling
ax.set_xlabel('Neighborhood Size (k)', fontsize=14, fontweight='bold')
ax.set_ylabel('NDCG@10', fontsize=14, fontweight='bold')
ax.set_title('Comprehensive ItemCF Performance Comparison\nNormalization Methods, Similarity Metrics, and Hyperparameters', 
             fontsize=15, fontweight='bold', pad=20)

# Legend
ax.legend(loc='lower right', frameon=True, shadow=True, fontsize=11, 
          ncol=2, columnspacing=1, handlelength=2.5)

# Grid
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
ax.set_xticks(k_values)
ax.set_xlim(5, 65)

# Add best configuration annotation
best_idx = df['NDCG'].idxmax()
best_row = df.loc[best_idx]
ax.annotate(f'Best: NDCG={best_row["NDCG"]:.4f}\n(Z-Score+Cosine, k={int(best_row["top_k"])}, min_r={int(best_row["min_ratings"])})',
            xy=(best_row['top_k'], best_row['NDCG']),
            xytext=(best_row['top_k']-15, best_row['NDCG']+0.008),
            fontsize=11,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3', lw=2))

# Y-axis formatting
ax.set_ylim(0.05, 0.22)

plt.tight_layout()
plt.savefig('results/itemcf_comprehensive_comparison.svg', dpi=300, bbox_inches='tight')
plt.savefig('results/itemcf_comprehensive_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Comprehensive comparison plot saved")

# ===== BONUS: Create a heatmap for best k per configuration =====
fig2, ax2 = plt.subplots(figsize=(12, 6))

# Prepare data for heatmap
methods = ['Z-Score\n+ Cosine', 'Mean-Center\n+ Cosine', 'Mean-Center\n+ Pearson']
min_ratings_vals = [3, 5, 10, 15, 20]

heatmap_data = []
for method_idx, (norm, sim) in enumerate([('zscore', 'cosine'), ('mean_center', 'cosine'), ('mean_center', 'pearson')]):
    row = []
    for min_r in min_ratings_vals:
        subset = df[(df['normalization'] == norm) & 
                   (df['similarity'] == sim) & 
                   (df['min_ratings'] == min_r)]
        if len(subset) > 0:
            best_ndcg = subset['NDCG'].max()
            row.append(best_ndcg)
        else:
            row.append(0)
    heatmap_data.append(row)

heatmap_data = np.array(heatmap_data)

# Create heatmap
im = ax2.imshow(heatmap_data, cmap='YlOrRd', aspect='auto', vmin=0.05, vmax=0.22)

# Set ticks and labels
ax2.set_xticks(np.arange(len(min_ratings_vals)))
ax2.set_yticks(np.arange(len(methods)))
ax2.set_xticklabels(min_ratings_vals, fontsize=12)
ax2.set_yticklabels(methods, fontsize=12)

# Add text annotations
for i in range(len(methods)):
    for j in range(len(min_ratings_vals)):
        text = ax2.text(j, i, f'{heatmap_data[i, j]:.3f}',
                       ha="center", va="center", color="black", fontsize=11, fontweight='bold')

ax2.set_xlabel('Minimum Ratings Filter', fontsize=13, fontweight='bold')
ax2.set_ylabel('Method', fontsize=13, fontweight='bold')
ax2.set_title('Best NDCG@10 Scores by Method and Min Ratings\n(Optimized k for each configuration)', 
             fontsize=14, fontweight='bold', pad=15)

# Colorbar
cbar = plt.colorbar(im, ax=ax2)
cbar.set_label('NDCG@10', rotation=270, labelpad=20, fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('results/itemcf_heatmap_comparison.svg', dpi=300, bbox_inches='tight')
plt.savefig('results/itemcf_heatmap_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Heatmap comparison saved")

# ===== Print Summary Statistics =====
print("\n" + "="*60)
print("ITEMCF COMPREHENSIVE ANALYSIS SUMMARY")
print("="*60)

print("\n📊 TOP 5 CONFIGURATIONS:")
top5 = df.nlargest(5, 'NDCG')[['normalization', 'similarity', 'min_ratings', 'top_k', 'NDCG']]
for idx, (i, row) in enumerate(top5.iterrows(), 1):
    print(f"{idx}. {row['normalization']:12s} + {row['similarity']:8s} | min_r={int(row['min_ratings']):2d} | k={int(row['top_k']):2d} | NDCG={row['NDCG']:.4f}")

print("\n📈 PERFORMANCE BY METHOD (Best NDCG@10):")
for norm, sim in [('zscore', 'cosine'), ('mean_center', 'cosine'), ('mean_center', 'pearson')]:
    subset = df[(df['normalization'] == norm) & (df['similarity'] == sim)]
    if len(subset) > 0:
        best = subset.loc[subset['NDCG'].idxmax()]
        print(f"  {norm:12s} + {sim:8s}: {best['NDCG']:.4f} (k={int(best['top_k'])}, min_r={int(best['min_ratings'])})")

print("\n🎯 KEY INSIGHTS:")
zscore_best = df[(df['normalization'] == 'zscore') & (df['similarity'] == 'cosine')]['NDCG'].max()
mean_cosine_best = df[(df['normalization'] == 'mean_center') & (df['similarity'] == 'cosine')]['NDCG'].max()
mean_pearson_best = df[(df['normalization'] == 'mean_center') & (df['similarity'] == 'pearson')]['NDCG'].max()

improvement_vs_mean_cosine = ((zscore_best - mean_cosine_best) / mean_cosine_best) * 100
improvement_vs_pearson = ((zscore_best - mean_pearson_best) / mean_pearson_best) * 100

print(f"  • Z-Score+Cosine vs Mean-Center+Cosine: +{improvement_vs_mean_cosine:.1f}% improvement")
print(f"  • Z-Score+Cosine vs Mean-Center+Pearson: +{improvement_vs_pearson:.1f}% improvement")
print(f"  • Cosine vs Pearson (Mean-Center): +{((mean_cosine_best - mean_pearson_best) / mean_pearson_best) * 100:.1f}% improvement")

print("\n✅ All visualizations created successfully!")
print("="*60)
