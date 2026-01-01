"""
Hybrid Model 1 Optimization Results Visualization
Visualizes the impact of C parameter and aggregation strategies on Group NDCG@10
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Read the results
results_df = pd.read_csv('results/hybrid1_optimized_results.csv')

# Create figure with multiple subplots
fig = plt.figure(figsize=(16, 10))

# ============================================================
# Plot 1: Line plot - C parameter impact by aggregation strategy
# ============================================================
ax1 = plt.subplot(2, 2, 1)
for agg in results_df['Aggregation'].unique():
    data = results_df[results_df['Aggregation'] == agg]
    ax1.plot(data['C'], data['NDCG@10'], marker='o', linewidth=2.5, 
             markersize=8, label=agg.replace('_', ' ').title())

ax1.set_xlabel('C Parameter (Trust Factor)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Group NDCG@10', fontsize=12, fontweight='bold')
ax1.set_title('Impact of C Parameter on Group NDCG@10\nby Aggregation Strategy', 
              fontsize=14, fontweight='bold', pad=20)
ax1.legend(title='Aggregation Strategy', fontsize=10, title_fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_xscale('log')

# Highlight best configuration
best_idx = results_df['NDCG@10'].idxmax()
best_c = results_df.loc[best_idx, 'C']
best_ndcg = results_df.loc[best_idx, 'NDCG@10']
best_agg = results_df.loc[best_idx, 'Aggregation']
ax1.scatter([best_c], [best_ndcg], s=200, c='red', marker='*', 
            zorder=5, edgecolors='darkred', linewidth=2,
            label=f'Best: C={best_c}, {best_agg}')

# ============================================================
# Plot 2: Grouped bar chart - Aggregation strategies comparison
# ============================================================
ax2 = plt.subplot(2, 2, 2)
c_values = results_df['C'].unique()
x = np.arange(len(c_values))
width = 0.25

aggregations = results_df['Aggregation'].unique()
colors = ['#3498db', '#e74c3c', '#2ecc71']

for i, agg in enumerate(aggregations):
    data = results_df[results_df['Aggregation'] == agg]
    ndcg_values = [data[data['C'] == c]['NDCG@10'].values[0] for c in c_values]
    ax2.bar(x + i*width, ndcg_values, width, label=agg.replace('_', ' ').title(),
            color=colors[i], alpha=0.8, edgecolor='black', linewidth=1.2)

ax2.set_xlabel('C Parameter (Trust Factor)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Group NDCG@10', fontsize=12, fontweight='bold')
ax2.set_title('Aggregation Strategy Comparison\nacross Different C Values', 
              fontsize=14, fontweight='bold', pad=20)
ax2.set_xticks(x + width)
ax2.set_xticklabels([f'{c}' for c in c_values])
ax2.legend(title='Aggregation Strategy', fontsize=10, title_fontsize=11)
ax2.grid(True, alpha=0.3, axis='y')

# ============================================================
# Plot 3: Heatmap - C vs Aggregation Strategy
# ============================================================
ax3 = plt.subplot(2, 2, 3)
pivot_table = results_df.pivot(index='Aggregation', columns='C', values='NDCG@10')
sns.heatmap(pivot_table, annot=True, fmt='.4f', cmap='YlOrRd', 
            cbar_kws={'label': 'Group NDCG@10'}, ax=ax3,
            linewidths=1, linecolor='white')
ax3.set_title('Performance Heatmap: C Parameter × Aggregation Strategy', 
              fontsize=14, fontweight='bold', pad=20)
ax3.set_xlabel('C Parameter (Trust Factor)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Aggregation Strategy', fontsize=12, fontweight='bold')
ax3.set_yticklabels([label.get_text().replace('_', ' ').title() 
                      for label in ax3.get_yticklabels()], rotation=0)

# ============================================================
# Plot 4: Performance ranking - All configurations
# ============================================================
ax4 = plt.subplot(2, 2, 4)
results_sorted = results_df.sort_values('NDCG@10', ascending=True)
results_sorted['Config'] = results_sorted.apply(
    lambda row: f"C={row['C']}, {row['Aggregation'][:3].upper()}", axis=1
)

colors_ranked = ['red' if i == len(results_sorted)-1 else 'steelblue' 
                 for i in range(len(results_sorted))]
bars = ax4.barh(range(len(results_sorted)), results_sorted['NDCG@10'], 
                color=colors_ranked, alpha=0.8, edgecolor='black', linewidth=1.2)

ax4.set_yticks(range(len(results_sorted)))
ax4.set_yticklabels(results_sorted['Config'], fontsize=9)
ax4.set_xlabel('Group NDCG@10', fontsize=12, fontweight='bold')
ax4.set_title('Configuration Ranking\n(Best to Worst)', 
              fontsize=14, fontweight='bold', pad=20)
ax4.grid(True, alpha=0.3, axis='x')

# Add value labels on bars
for i, (idx, row) in enumerate(results_sorted.iterrows()):
    ax4.text(row['NDCG@10'] + 0.005, i, f"{row['NDCG@10']:.4f}", 
             va='center', fontsize=8, fontweight='bold')

# ============================================================
# Overall title and layout
# ============================================================
fig.suptitle('Hybrid Model 1: Comprehensive Optimization Analysis\nGroup NDCG@10 Performance', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.98])

# Save the figure
plt.savefig('results/hybrid1_optimization_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: results/hybrid1_optimization_analysis.png")

# ============================================================
# Additional Analysis: Statistical Summary
# ============================================================
print("\n" + "="*60)
print("HYBRID MODEL 1 - OPTIMIZATION RESULTS SUMMARY")
print("="*60)

print(f"\n📊 Best Configuration:")
print(f"   C Parameter: {best_c}")
print(f"   Aggregation: {best_agg.replace('_', ' ').title()}")
print(f"   Group NDCG@10: {best_ndcg:.6f}")

print(f"\n📈 Performance by Aggregation Strategy:")
for agg in results_df['Aggregation'].unique():
    agg_data = results_df[results_df['Aggregation'] == agg]
    print(f"\n   {agg.replace('_', ' ').title()}:")
    print(f"      Best C: {agg_data.loc[agg_data['NDCG@10'].idxmax(), 'C']}")
    print(f"      Best NDCG@10: {agg_data['NDCG@10'].max():.6f}")
    print(f"      Worst NDCG@10: {agg_data['NDCG@10'].min():.6f}")
    print(f"      Range: {agg_data['NDCG@10'].max() - agg_data['NDCG@10'].min():.6f}")

print(f"\n📉 Performance by C Parameter:")
for c in sorted(results_df['C'].unique()):
    c_data = results_df[results_df['C'] == c]
    print(f"\n   C = {c}:")
    print(f"      Best Aggregation: {c_data.loc[c_data['NDCG@10'].idxmax(), 'Aggregation'].replace('_', ' ').title()}")
    print(f"      Best NDCG@10: {c_data['NDCG@10'].max():.6f}")
    print(f"      Average NDCG@10: {c_data['NDCG@10'].mean():.6f}")

print("\n" + "="*60)

# Show the plot
plt.show()
