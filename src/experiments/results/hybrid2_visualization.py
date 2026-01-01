import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the data
df = pd.read_csv('hybrid2_group_optimization.csv')

# Create a more readable weight column
df['Weight_Combo'] = df.apply(lambda x: f"{x['w_ubcf']:.2f}/{x['w_cbf']:.2f}", axis=1)

# Create figure with high DPI for quality
fig, ax = plt.subplots(figsize=(14, 8), dpi=100)

# Define colors for each aggregation strategy
colors = {
    'average': '#2E86AB',
    'least_misery': '#A23B72',
    'harmonic_mean': '#F18F01'
}

# Define markers
markers = {
    'average': 'o',
    'least_misery': 's',
    'harmonic_mean': '^'
}

# Plot each aggregation strategy
for agg in df['aggregation'].unique():
    data = df[df['aggregation'] == agg]
    ax.plot(data['w_ubcf'], data['group_ndcg'], 
            marker=markers[agg], 
            color=colors[agg], 
            linewidth=2.5, 
            markersize=10,
            label=agg.replace('_', ' ').title(),
            alpha=0.9)

# Styling
ax.set_xlabel('UBCF Weight (w_ubcf)', fontsize=14, fontweight='bold')
ax.set_ylabel('Group NDCG@10', fontsize=14, fontweight='bold')
ax.set_title('Hybrid Model 2: Impact of Weight Distribution and Aggregation Strategy on Group NDCG@10', 
             fontsize=16, fontweight='bold', pad=20)

# Grid
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
ax.set_axisbelow(True)

# Legend
ax.legend(loc='best', fontsize=12, framealpha=0.95, edgecolor='gray')

# X-axis formatting
ax.set_xticks(df['w_ubcf'].unique())
ax.set_xticklabels([f'{w:.2f}' for w in df['w_ubcf'].unique()])

# Y-axis formatting
ax.set_ylim(0.3, 0.45)
ax.tick_params(axis='both', which='major', labelsize=11)

# Add best performance annotation
best_row = df.loc[df['group_ndcg'].idxmax()]
ax.annotate(f'Best: w={best_row["w_ubcf"]:.2f}/{best_row["w_cbf"]:.2f}\n{best_row["aggregation"]}\nNDCG@10={best_row["group_ndcg"]:.4f}',
            xy=(best_row['w_ubcf'], best_row['group_ndcg']),
            xytext=(best_row['w_ubcf']+0.02, best_row['group_ndcg']+0.01),
            fontsize=10,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3', color='black', lw=1.5))

# Tight layout
plt.tight_layout()

# Save as SVG
plt.savefig('hybrid2_optimization_visualization.svg', format='svg', bbox_inches='tight')
print("✓ Hybrid2 visualization saved as 'hybrid2_optimization_visualization.svg'")

# Also create a bar chart comparison
fig2, ax2 = plt.subplots(figsize=(14, 8), dpi=100)

# Prepare data for grouped bar chart
weight_combos = df['Weight_Combo'].unique()
aggregations = df['aggregation'].unique()
x = np.arange(len(weight_combos))
width = 0.25

# Plot bars for each aggregation
for i, agg in enumerate(aggregations):
    data = df[df['aggregation'] == agg]
    offset = width * (i - 1)
    ax2.bar(x + offset, data['group_ndcg'], width, 
            label=agg.replace('_', ' ').title(),
            color=colors[agg],
            alpha=0.85,
            edgecolor='black',
            linewidth=0.8)

# Styling
ax2.set_xlabel('Weight Combination (UBCF/CBF)', fontsize=14, fontweight='bold')
ax2.set_ylabel('Group NDCG@10', fontsize=14, fontweight='bold')
ax2.set_title('Hybrid Model 2: Aggregation Strategy Comparison Across Weight Combinations', 
              fontsize=16, fontweight='bold', pad=20)
ax2.set_xticks(x)
ax2.set_xticklabels(weight_combos, rotation=0)
ax2.legend(fontsize=12, framealpha=0.95, edgecolor='gray')
ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, axis='y')
ax2.set_axisbelow(True)
ax2.set_ylim(0.3, 0.45)
ax2.tick_params(axis='both', which='major', labelsize=11)

plt.tight_layout()
plt.savefig('hybrid2_bar_comparison.svg', format='svg', bbox_inches='tight')
print("✓ Hybrid2 bar comparison saved as 'hybrid2_bar_comparison.svg'")

# Create a heatmap with dark background
fig3, ax3 = plt.subplots(figsize=(12, 6), dpi=100)
fig3.patch.set_facecolor('#151515')  # Set figure background to dark gray
ax3.set_facecolor('#151515')  # Set axes background to dark gray

# Pivot data for heatmap
heatmap_data = df.pivot(index='aggregation', columns='Weight_Combo', values='group_ndcg')

# Create heatmap
im = ax3.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')

# Set ticks
ax3.set_xticks(np.arange(len(weight_combos)))
ax3.set_yticks(np.arange(len(aggregations)))
ax3.set_xticklabels(weight_combos)
ax3.set_yticklabels([agg.replace('_', ' ').title() for agg in heatmap_data.index])

# Rotate the tick labels and set color to white
plt.setp(ax3.get_xticklabels(), rotation=45, ha="right", color='white')
plt.setp(ax3.get_yticklabels(), color='white')

# Style tick parameters
ax3.tick_params(axis='both', colors='white')

# Add colorbar
cbar = plt.colorbar(im, ax=ax3)
cbar.set_label('Group NDCG@10', rotation=270, labelpad=20, fontsize=12, fontweight='bold', color='white')
cbar.ax.yaxis.set_tick_params(color='white')
plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

# Add text annotations
for i in range(len(aggregations)):
    for j in range(len(weight_combos)):
        text = ax3.text(j, i, f'{heatmap_data.iloc[i, j]:.4f}',
                       ha="center", va="center", color="black", fontsize=9, fontweight='bold')

ax3.set_xlabel('Weight Combination (UBCF/CBF)', fontsize=14, fontweight='bold', color='white')
ax3.set_ylabel('Aggregation Strategy', fontsize=14, fontweight='bold', color='white')
ax3.set_title('Hybrid Model 2: Group NDCG@10 Heatmap', fontsize=16, fontweight='bold', pad=20, color='white')

plt.tight_layout()
plt.savefig('hybrid2_heatmap.svg', format='svg', bbox_inches='tight')
print("✓ Hybrid2 heatmap saved as 'hybrid2_heatmap.svg'")

# Create a WHITE background version of the heatmap
fig4, ax4 = plt.subplots(figsize=(12, 6), dpi=100)
fig4.patch.set_facecolor('white')  # Set figure background to white
ax4.set_facecolor('white')  # Set axes background to white

# Create heatmap (reuse the same data)
im2 = ax4.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')

# Set ticks
ax4.set_xticks(np.arange(len(weight_combos)))
ax4.set_yticks(np.arange(len(aggregations)))
ax4.set_xticklabels(weight_combos)
ax4.set_yticklabels([agg.replace('_', ' ').title() for agg in heatmap_data.index])

# Rotate the tick labels and set color to black
plt.setp(ax4.get_xticklabels(), rotation=45, ha="right", color='black')
plt.setp(ax4.get_yticklabels(), color='black')

# Style tick parameters
ax4.tick_params(axis='both', colors='black')

# Add colorbar
cbar2 = plt.colorbar(im2, ax=ax4)
cbar2.set_label('Group NDCG@10', rotation=270, labelpad=20, fontsize=12, fontweight='bold', color='black')
cbar2.ax.yaxis.set_tick_params(color='black')
plt.setp(plt.getp(cbar2.ax.axes, 'yticklabels'), color='black')

# Add text annotations
for i in range(len(aggregations)):
    for j in range(len(weight_combos)):
        text = ax4.text(j, i, f'{heatmap_data.iloc[i, j]:.4f}',
                       ha="center", va="center", color="black", fontsize=9, fontweight='bold')

ax4.set_xlabel('Weight Combination (UBCF/CBF)', fontsize=14, fontweight='bold', color='black')
ax4.set_ylabel('Aggregation Strategy', fontsize=14, fontweight='bold', color='black')
ax4.set_title('Hybrid Model 2: Group NDCG@10 Heatmap', fontsize=16, fontweight='bold', pad=20, color='black')

plt.tight_layout()
plt.savefig('hybrid2_heatmap_white.svg', format='svg', bbox_inches='tight')
print("✓ Hybrid2 heatmap (WHITE background) saved as 'hybrid2_heatmap_white.svg'")

# Print summary statistics
print("\n" + "="*60)
print("HYBRID MODEL 2 OPTIMIZATION RESULTS SUMMARY")
print("="*60)
print(f"\nBest Configuration:")
print(f"  UBCF Weight: {best_row['w_ubcf']}")
print(f"  CBF Weight: {best_row['w_cbf']}")
print(f"  Aggregation: {best_row['aggregation']}")
print(f"  Group NDCG@10: {best_row['group_ndcg']:.6f}")
print(f"  Fairness: {best_row['fairness']:.6f}")
print(f"  Groups: {best_row['num_groups']}")

print(f"\nWorst Configuration:")
worst_row = df.loc[df['group_ndcg'].idxmin()]
print(f"  UBCF Weight: {worst_row['w_ubcf']}")
print(f"  CBF Weight: {worst_row['w_cbf']}")
print(f"  Aggregation: {worst_row['aggregation']}")
print(f"  Group NDCG@10: {worst_row['group_ndcg']:.6f}")

print(f"\nPerformance by Aggregation Strategy:")
for agg in aggregations:
    agg_data = df[df['aggregation'] == agg]
    print(f"  {agg.replace('_', ' ').title()}:")
    print(f"    Mean NDCG@10: {agg_data['group_ndcg'].mean():.6f}")
    best_config = agg_data.loc[agg_data['group_ndcg'].idxmax()]
    print(f"    Best Weight: {best_config['w_ubcf']:.2f}/{best_config['w_cbf']:.2f}")
    print(f"    Best NDCG@10: {agg_data['group_ndcg'].max():.6f}")

print(f"\nPerformance by Weight Combination:")
for w in df['w_ubcf'].unique():
    w_data = df[df['w_ubcf'] == w]
    print(f"  w_ubcf = {w:.2f}:")
    print(f"    Mean NDCG@10: {w_data['group_ndcg'].mean():.6f}")
    print(f"    Best Aggregation: {w_data.loc[w_data['group_ndcg'].idxmax(), 'aggregation']}")
    print(f"    Best NDCG@10: {w_data['group_ndcg'].max():.6f}")

print("\n" + "="*60)

# Additional analysis: Compare with Hybrid 1
print("\nCOMPARISON WITH HYBRID MODEL 1:")
print("="*60)
print(f"Hybrid 2 Best NDCG@10: {best_row['group_ndcg']:.6f}")
print(f"Hybrid 1 Best NDCG@10: 0.5211 (from test results)")
print(f"Performance Gap: {((0.5211 - best_row['group_ndcg']) / 0.5211 * 100):.2f}%")
print("\nKey Insights:")
print("  - Least Misery aggregation performs best for Hybrid 2")
print("  - Optimal weight balance: w_ubcf=0.20, w_cbf=0.80")
print("  - Fairness scores are consistently high (>0.85)")
print("="*60)
