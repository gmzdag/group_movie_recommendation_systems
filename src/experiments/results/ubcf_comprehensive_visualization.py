import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Read the NDCG data
df = pd.read_csv('ubcf_grid_search_ndcg.csv')

print("="*60)
print("UBCF GRID SEARCH RESULTS")
print("="*60)
print(f"\nTotal configurations: {len(df)}")
print(f"\nBest configuration:")
best_idx = df['NDCG@10'].idxmax()
best_row = df.loc[best_idx]
print(f"  Method: {best_row['Method']}")
print(f"  K_Neighbors: {best_row['K_Neighbors']}")
print(f"  Min_Overlap: {best_row['Min_Overlap']}")
print(f"  NDCG@10: {best_row['NDCG@10']:.6f}")

print(f"\nWorst configuration:")
worst_idx = df['NDCG@10'].idxmin()
worst_row = df.loc[worst_idx]
print(f"  Method: {worst_row['Method']}")
print(f"  K_Neighbors: {worst_row['K_Neighbors']}")
print(f"  Min_Overlap: {worst_row['Min_Overlap']}")
print(f"  NDCG@10: {worst_row['NDCG@10']:.6f}")

# ============================================================================
# FIGURE 1: K Parameter Sensitivity (similar to IBCF fig_2)
# ============================================================================
fig1, ax1 = plt.subplots(figsize=(12, 7), dpi=100)
fig1.patch.set_facecolor('white')
ax1.set_facecolor('white')

# Define colors for each method
method_colors = {
    'Cosine': '#2E86AB',
    'Pearson_SW': '#A23B72',
    'Pearson_Shrink': '#F18F01',
    'Spearman_Rank': '#06A77D',
    'Spearman_SW': '#D62828'
}

markers = {
    'Cosine': 'o',
    'Pearson_SW': 's',
    'Pearson_Shrink': '^',
    'Spearman_Rank': 'D',
    'Spearman_SW': 'v'
}

# Plot for each method and min_overlap
for method in df['Method'].unique():
    for min_overlap in df['Min_Overlap'].unique():
        data = df[(df['Method'] == method) & (df['Min_Overlap'] == min_overlap)]
        data = data.sort_values('K_Neighbors')
        
        label = f'{method} (overlap={min_overlap})'
        linestyle = '-' if min_overlap == 5 else '--'
        alpha = 0.9 if min_overlap == 5 else 0.6
        
        ax1.plot(data['K_Neighbors'], data['NDCG@10'],
                marker=markers[method],
                color=method_colors[method],
                linewidth=2.5 if min_overlap == 5 else 1.5,
                markersize=10 if min_overlap == 5 else 7,
                label=label,
                linestyle=linestyle,
                alpha=alpha)

ax1.set_xlabel('K (Number of Neighbors)', fontsize=14, fontweight='bold', color='black')
ax1.set_ylabel('NDCG@10', fontsize=14, fontweight='bold', color='black')
ax1.set_title('UBCF: Impact of K Parameter on NDCG@10 Performance', 
             fontsize=16, fontweight='bold', pad=20, color='black')

ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
ax1.set_axisbelow(True)
ax1.legend(loc='best', fontsize=10, framealpha=0.95, edgecolor='gray', ncol=2)
ax1.set_xticks([20, 50])
ax1.set_xticklabels(['20', '50'])
ax1.tick_params(axis='both', which='major', labelsize=11, colors='black')

# Add best point annotation
ax1.annotate(f'Best: Cosine, K=20, overlap=5\nNDCG@10={best_row["NDCG@10"]:.4f}',
            xy=(best_row['K_Neighbors'], best_row['NDCG@10']),
            xytext=(best_row['K_Neighbors']+5, best_row['NDCG@10']-0.05),
            fontsize=10,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3', color='black', lw=1.5))

plt.tight_layout()
plt.savefig('ubcf_k_sensitivity_clean.svg', format='svg', bbox_inches='tight')
print("\n✓ Saved: ubcf_k_sensitivity_clean.svg")

# ============================================================================
# FIGURE 2: Min_Overlap Parameter Sensitivity (similar to IBCF fig_3)
# ============================================================================
fig2, ax2 = plt.subplots(figsize=(12, 7), dpi=100)
fig2.patch.set_facecolor('white')
ax2.set_facecolor('white')

# Plot for each method and K
for method in df['Method'].unique():
    for k in df['K_Neighbors'].unique():
        data = df[(df['Method'] == method) & (df['K_Neighbors'] == k)]
        data = data.sort_values('Min_Overlap')
        
        label = f'{method} (K={k})'
        linestyle = '-' if k == 20 else '--'
        alpha = 0.9 if k == 20 else 0.6
        
        ax2.plot(data['Min_Overlap'], data['NDCG@10'],
                marker=markers[method],
                color=method_colors[method],
                linewidth=2.5 if k == 20 else 1.5,
                markersize=10 if k == 20 else 7,
                label=label,
                linestyle=linestyle,
                alpha=alpha)

ax2.set_xlabel('Min Overlap (Minimum Common Ratings)', fontsize=14, fontweight='bold', color='black')
ax2.set_ylabel('NDCG@10', fontsize=14, fontweight='bold', color='black')
ax2.set_title('UBCF: Impact of Min Overlap Parameter on NDCG@10 Performance', 
             fontsize=16, fontweight='bold', pad=20, color='black')

ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
ax2.set_axisbelow(True)
ax2.legend(loc='best', fontsize=10, framealpha=0.95, edgecolor='gray', ncol=2)
ax2.set_xticks([5, 10])
ax2.set_xticklabels(['5', '10'])
ax2.tick_params(axis='both', which='major', labelsize=11, colors='black')

plt.tight_layout()
plt.savefig('ubcf_overlap_sensitivity_clean.svg', format='svg', bbox_inches='tight')
print("✓ Saved: ubcf_overlap_sensitivity_clean.svg")

# ============================================================================
# FIGURE 3: Method Comparison Bar Chart
# ============================================================================
fig3, ax3 = plt.subplots(figsize=(14, 8), dpi=100)
fig3.patch.set_facecolor('white')
ax3.set_facecolor('white')

# Prepare data for grouped bar chart
methods = df['Method'].unique()
k_values = df['K_Neighbors'].unique()
overlap_values = df['Min_Overlap'].unique()

# Create a configuration label
df['Config'] = df.apply(lambda x: f"K={int(x['K_Neighbors'])}, O={int(x['Min_Overlap'])}", axis=1)
configs = df['Config'].unique()

x = np.arange(len(methods))
width = 0.2

# Plot bars for each configuration
for i, config in enumerate(configs):
    data = df[df['Config'] == config].sort_values('Method')
    offset = width * (i - 1.5)
    ax3.bar(x + offset, data['NDCG@10'], width,
            label=config,
            alpha=0.85,
            edgecolor='black',
            linewidth=0.8)

ax3.set_xlabel('Similarity Method', fontsize=14, fontweight='bold', color='black')
ax3.set_ylabel('NDCG@10', fontsize=14, fontweight='bold', color='black')
ax3.set_title('UBCF: Similarity Method Comparison Across Configurations', 
              fontsize=16, fontweight='bold', pad=20, color='black')
ax3.set_xticks(x)
ax3.set_xticklabels(methods, rotation=15, ha='right')
ax3.legend(fontsize=11, framealpha=0.95, edgecolor='gray', title='Configuration')
ax3.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, axis='y')
ax3.set_axisbelow(True)
ax3.tick_params(axis='both', which='major', labelsize=11, colors='black')

plt.tight_layout()
plt.savefig('ubcf_method_comparison_clean.svg', format='svg', bbox_inches='tight')
print("✓ Saved: ubcf_method_comparison_clean.svg")

# ============================================================================
# FIGURE 4: Heatmap (similar to existing ubcf_heatmap_white.svg)
# ============================================================================
fig4, ax4 = plt.subplots(figsize=(14, 8), dpi=100)
fig4.patch.set_facecolor('white')
ax4.set_facecolor('white')

# Create a pivot table for heatmap
# We'll create a multi-index heatmap: Method × (K, Overlap)
df['K_Overlap'] = df.apply(lambda x: f"K={int(x['K_Neighbors'])}\nO={int(x['Min_Overlap'])}", axis=1)
heatmap_data = df.pivot(index='Method', columns='K_Overlap', values='NDCG@10')

# Create heatmap
im = ax4.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0.2, vmax=0.45)

# Set ticks
ax4.set_xticks(np.arange(len(heatmap_data.columns)))
ax4.set_yticks(np.arange(len(heatmap_data.index)))
ax4.set_xticklabels(heatmap_data.columns)
ax4.set_yticklabels(heatmap_data.index)

# Rotate the tick labels
plt.setp(ax4.get_xticklabels(), rotation=0, ha="center", color='black', fontsize=10)
plt.setp(ax4.get_yticklabels(), color='black', fontsize=11)

# Style tick parameters
ax4.tick_params(axis='both', colors='black')

# Add colorbar
cbar = plt.colorbar(im, ax=ax4)
cbar.set_label('NDCG@10', rotation=270, labelpad=20, fontsize=12, fontweight='bold', color='black')
cbar.ax.yaxis.set_tick_params(color='black')
plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='black')

# Add text annotations
for i in range(len(heatmap_data.index)):
    for j in range(len(heatmap_data.columns)):
        value = heatmap_data.iloc[i, j]
        text_color = 'white' if value < 0.3 else 'black'
        text = ax4.text(j, i, f'{value:.4f}',
                       ha="center", va="center", color=text_color, fontsize=9, fontweight='bold')

ax4.set_xlabel('Configuration (K, Min_Overlap)', fontsize=14, fontweight='bold', color='black')
ax4.set_ylabel('Similarity Method', fontsize=14, fontweight='bold', color='black')
ax4.set_title('UBCF: NDCG@10 Heatmap Across All Configurations', 
              fontsize=16, fontweight='bold', pad=20, color='black')

plt.tight_layout()
plt.savefig('ubcf_comprehensive_heatmap_white.svg', format='svg', bbox_inches='tight')
print("✓ Saved: ubcf_comprehensive_heatmap_white.svg")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)

print("\nPerformance by Method:")
for method in methods:
    method_data = df[df['Method'] == method]
    print(f"\n{method}:")
    print(f"  Mean NDCG@10: {method_data['NDCG@10'].mean():.6f}")
    print(f"  Best NDCG@10: {method_data['NDCG@10'].max():.6f}")
    best_config = method_data.loc[method_data['NDCG@10'].idxmax()]
    print(f"  Best Config: K={int(best_config['K_Neighbors'])}, Overlap={int(best_config['Min_Overlap'])}")

print("\nPerformance by K:")
for k in k_values:
    k_data = df[df['K_Neighbors'] == k]
    print(f"\nK={int(k)}:")
    print(f"  Mean NDCG@10: {k_data['NDCG@10'].mean():.6f}")
    print(f"  Best NDCG@10: {k_data['NDCG@10'].max():.6f}")

print("\nPerformance by Min_Overlap:")
for overlap in overlap_values:
    overlap_data = df[df['Min_Overlap'] == overlap]
    print(f"\nMin_Overlap={int(overlap)}:")
    print(f"  Mean NDCG@10: {overlap_data['NDCG@10'].mean():.6f}")
    print(f"  Best NDCG@10: {overlap_data['NDCG@10'].max():.6f}")

print("\n" + "="*60)
print("KEY INSIGHTS")
print("="*60)
print("\n1. Cosine similarity significantly outperforms other methods")
print(f"   - Cosine best: {df[df['Method']=='Cosine']['NDCG@10'].max():.4f}")
print(f"   - Others best: {df[df['Method']!='Cosine']['NDCG@10'].max():.4f}")
print(f"   - Improvement: {((df[df['Method']=='Cosine']['NDCG@10'].max() / df[df['Method']!='Cosine']['NDCG@10'].max() - 1) * 100):.1f}%")

print("\n2. K=20 generally outperforms K=50")
print(f"   - K=20 mean: {df[df['K_Neighbors']==20]['NDCG@10'].mean():.4f}")
print(f"   - K=50 mean: {df[df['K_Neighbors']==50]['NDCG@10'].mean():.4f}")

print("\n3. Min_Overlap=5 provides better performance")
print(f"   - Overlap=5 mean: {df[df['Min_Overlap']==5]['NDCG@10'].mean():.4f}")
print(f"   - Overlap=10 mean: {df[df['Min_Overlap']==10]['NDCG@10'].mean():.4f}")

print("\n" + "="*60)
print("ALL VISUALIZATIONS CREATED SUCCESSFULLY!")
print("="*60)
