import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the comprehensive optimization data
df_comp = pd.read_csv('../watchlist/watchlist_comprehensive_optimization.csv')

# Read the master group data
df_master = pd.read_csv('../watchlist/watchlist_master_group.csv')

print("="*60)
print("WATCHLIST OPTIMIZATION RESULTS")
print("="*60)
print(f"\nComprehensive optimization: {len(df_comp)} configurations")
print(f"Master group results: {len(df_master)} configurations")

# ============================================================================
# FIGURE 1: Comprehensive Optimization Heatmap (like Hybrid 1 & 2)
# ============================================================================
fig1, ax1 = plt.subplots(figsize=(14, 8), dpi=100)
fig1.patch.set_facecolor('white')
ax1.set_facecolor('white')

# Prepare data for heatmap
# Create a readable label for each configuration
df_comp['Config'] = df_comp.apply(
    lambda x: f"{x['model_type']}\n{x['aggregation']}\nPenalty={x['disagreement_penalty']}", 
    axis=1
)

# Create heatmap data
heatmap_data = df_comp.pivot_table(
    index='aggregation',
    columns='model_type',
    values='group_ndcg',
    aggfunc='mean'
)

# Create heatmap
im1 = ax1.imshow(heatmap_data, cmap='YlOrRd', aspect='auto', vmin=0.4, vmax=1.0)

# Set ticks
ax1.set_xticks(np.arange(len(heatmap_data.columns)))
ax1.set_yticks(np.arange(len(heatmap_data.index)))
ax1.set_xticklabels(heatmap_data.columns, rotation=15, ha='right')
ax1.set_yticklabels(heatmap_data.index)

# Style tick parameters
plt.setp(ax1.get_xticklabels(), color='black', fontsize=10)
plt.setp(ax1.get_yticklabels(), color='black', fontsize=11)
ax1.tick_params(axis='both', colors='black')

# Add colorbar
cbar1 = plt.colorbar(im1, ax=ax1)
cbar1.set_label('Group NDCG@10', rotation=270, labelpad=20, fontsize=12, fontweight='bold', color='black')
cbar1.ax.yaxis.set_tick_params(color='black')
plt.setp(plt.getp(cbar1.ax.axes, 'yticklabels'), color='black')

# Add text annotations
for i in range(len(heatmap_data.index)):
    for j in range(len(heatmap_data.columns)):
        value = heatmap_data.iloc[i, j]
        if not np.isnan(value):
            text_color = 'white' if value < 0.7 else 'black'
            text = ax1.text(j, i, f'{value:.3f}',
                           ha="center", va="center", color=text_color, 
                           fontsize=10, fontweight='bold')

ax1.set_xlabel('Model Type', fontsize=14, fontweight='bold', color='black')
ax1.set_ylabel('Aggregation Strategy', fontsize=14, fontweight='bold', color='black')
ax1.set_title('Watchlist-Based Filtering: Group NDCG@10 Heatmap', 
              fontsize=16, fontweight='bold', pad=20, color='black')

plt.tight_layout()
plt.savefig('../watchlist/watchlist_optimization_heatmap_white.svg', format='svg', bbox_inches='tight')
print("\n✓ Saved: watchlist_optimization_heatmap_white.svg")

# ============================================================================
# FIGURE 2: Strategy Comparison (like Hybrid 1 bar chart)
# ============================================================================
fig2, ax2 = plt.subplots(figsize=(14, 8), dpi=100)
fig2.patch.set_facecolor('white')
ax2.set_facecolor('white')

# Group by aggregation strategy
strategy_performance = df_master.groupby('Strategy').agg({
    'group_ndcg@10': 'mean',
    'group_fairness': 'mean',
    'num_groups': 'sum'
}).reset_index()

# Define colors
colors = {
    'Average': '#2E86AB',
    'Least Misery': '#A23B72',
    'Hybrid 70/30': '#F18F01'
}

x = np.arange(len(strategy_performance))
width = 0.35

# Plot NDCG bars
bars1 = ax2.bar(x - width/2, strategy_performance['group_ndcg@10'], width,
                label='Group NDCG@10', color='#2E86AB', alpha=0.85,
                edgecolor='black', linewidth=0.8)

# Plot Fairness bars
bars2 = ax2.bar(x + width/2, strategy_performance['group_fairness'], width,
                label='Group Fairness', color='#A23B72', alpha=0.85,
                edgecolor='black', linewidth=0.8)

ax2.set_xlabel('Aggregation Strategy', fontsize=14, fontweight='bold', color='black')
ax2.set_ylabel('Score', fontsize=14, fontweight='bold', color='black')
ax2.set_title('Watchlist: Aggregation Strategy Performance Comparison', 
              fontsize=16, fontweight='bold', pad=20, color='black')
ax2.set_xticks(x)
ax2.set_xticklabels(strategy_performance['Strategy'])
ax2.legend(fontsize=12, framealpha=0.95, edgecolor='gray')
ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, axis='y')
ax2.set_axisbelow(True)
ax2.tick_params(axis='both', which='major', labelsize=11, colors='black')

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('../watchlist/watchlist_strategy_comparison.svg', format='svg', bbox_inches='tight')
print("✓ Saved: watchlist_strategy_comparison.svg")

# ============================================================================
# FIGURE 3: Disagreement Penalty Impact (line plot)
# ============================================================================
fig3, ax3 = plt.subplots(figsize=(12, 7), dpi=100)
fig3.patch.set_facecolor('white')
ax3.set_facecolor('white')

# Filter for WatchlistRecommender with AVG_Profile
penalty_data = df_comp[
    (df_comp['model_type'] == 'WatchlistRecommender') & 
    (df_comp['aggregation'] == 'AVG_Profile')
].sort_values('disagreement_penalty')

if len(penalty_data) > 0:
    ax3.plot(penalty_data['disagreement_penalty'], penalty_data['group_ndcg'],
            marker='o', color='#2E86AB', linewidth=2.5, markersize=10,
            label='Group NDCG@10', alpha=0.9)
    
    ax3_twin = ax3.twinx()
    ax3_twin.plot(penalty_data['disagreement_penalty'], penalty_data['fairness'],
                 marker='s', color='#A23B72', linewidth=2.5, markersize=10,
                 label='Fairness', alpha=0.9)
    
    ax3.set_xlabel('Disagreement Penalty', fontsize=14, fontweight='bold', color='black')
    ax3.set_ylabel('Group NDCG@10', fontsize=14, fontweight='bold', color='#2E86AB')
    ax3_twin.set_ylabel('Fairness', fontsize=14, fontweight='bold', color='#A23B72')
    ax3.set_title('Watchlist: Impact of Disagreement Penalty on Performance', 
                 fontsize=16, fontweight='bold', pad=20, color='black')
    
    ax3.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax3.set_axisbelow(True)
    ax3.tick_params(axis='y', labelcolor='#2E86AB', colors='#2E86AB')
    ax3_twin.tick_params(axis='y', labelcolor='#A23B72', colors='#A23B72')
    ax3.tick_params(axis='x', colors='black')
    
    # Add legends
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=11)

plt.tight_layout()
plt.savefig('../watchlist/watchlist_penalty_impact.svg', format='svg', bbox_inches='tight')
print("✓ Saved: watchlist_penalty_impact.svg")

# ============================================================================
# FIGURE 4: Model Type Comparison (comprehensive)
# ============================================================================
fig4, ax4 = plt.subplots(figsize=(14, 8), dpi=100)
fig4.patch.set_facecolor('white')
ax4.set_facecolor('white')

# Group by model type
model_performance = df_comp.groupby('model_type').agg({
    'group_ndcg': ['mean', 'max', 'min'],
    'fairness': 'mean',
    'num_groups': 'sum'
}).reset_index()

model_performance.columns = ['model_type', 'ndcg_mean', 'ndcg_max', 'ndcg_min', 
                             'fairness_mean', 'num_groups']

x = np.arange(len(model_performance))
width = 0.25

# Plot bars
bars1 = ax4.bar(x - width, model_performance['ndcg_mean'], width,
                label='Mean NDCG@10', color='#2E86AB', alpha=0.85,
                edgecolor='black', linewidth=0.8)
bars2 = ax4.bar(x, model_performance['ndcg_max'], width,
                label='Max NDCG@10', color='#06A77D', alpha=0.85,
                edgecolor='black', linewidth=0.8)
bars3 = ax4.bar(x + width, model_performance['fairness_mean'], width,
                label='Mean Fairness', color='#A23B72', alpha=0.85,
                edgecolor='black', linewidth=0.8)

ax4.set_xlabel('Model Type', fontsize=14, fontweight='bold', color='black')
ax4.set_ylabel('Score', fontsize=14, fontweight='bold', color='black')
ax4.set_title('Watchlist: Model Type Performance Comparison', 
              fontsize=16, fontweight='bold', pad=20, color='black')
ax4.set_xticks(x)
ax4.set_xticklabels(model_performance['model_type'], rotation=15, ha='right')
ax4.legend(fontsize=11, framealpha=0.95, edgecolor='gray')
ax4.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, axis='y')
ax4.set_axisbelow(True)
ax4.tick_params(axis='both', which='major', labelsize=10, colors='black')

plt.tight_layout()
plt.savefig('../watchlist/watchlist_model_comparison.svg', format='svg', bbox_inches='tight')
print("✓ Saved: watchlist_model_comparison.svg")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================
print("\n" + "="*60)
print("WATCHLIST OPTIMIZATION SUMMARY")
print("="*60)

print("\nBest Configuration (Comprehensive):")
best_idx = df_comp['group_ndcg'].idxmax()
best_row = df_comp.loc[best_idx]
print(f"  Model Type: {best_row['model_type']}")
print(f"  Aggregation: {best_row['aggregation']}")
print(f"  Penalty: {best_row['disagreement_penalty']}")
print(f"  Group NDCG@10: {best_row['group_ndcg']:.6f}")
print(f"  Fairness: {best_row['fairness']:.6f}")
print(f"  Groups: {best_row['num_groups']}")

print("\nBest Configuration (Master Group):")
best_master_idx = df_master['group_ndcg@10'].idxmax()
best_master = df_master.loc[best_master_idx]
print(f"  Model: {best_master['Model']}")
print(f"  Strategy: {best_master['Strategy']}")
print(f"  Group Size: {best_master['Group_Size']}")
print(f"  Group NDCG@10: {best_master['group_ndcg@10']:.6f}")
print(f"  Fairness: {best_master['group_fairness']:.6f}")

print("\nPerformance by Strategy (Master Group):")
for strategy in df_master['Strategy'].unique():
    strategy_data = df_master[df_master['Strategy'] == strategy]
    print(f"\n{strategy}:")
    print(f"  Mean NDCG@10: {strategy_data['group_ndcg@10'].mean():.6f}")
    print(f"  Mean Fairness: {strategy_data['group_fairness'].mean():.6f}")
    print(f"  Total Groups: {strategy_data['num_groups'].sum()}")

print("\nPerformance by Model Type (Comprehensive):")
for model in df_comp['model_type'].unique():
    model_data = df_comp[df_comp['model_type'] == model]
    print(f"\n{model}:")
    print(f"  Mean NDCG@10: {model_data['group_ndcg'].mean():.6f}")
    print(f"  Max NDCG@10: {model_data['group_ndcg'].max():.6f}")
    print(f"  Mean Fairness: {model_data['fairness'].mean():.6f}")

print("\n" + "="*60)
print("KEY INSIGHTS")
print("="*60)
print("\n1. Hybrid3_Baseline achieves highest NDCG@10")
print(f"   - Best NDCG: {df_comp[df_comp['model_type']=='Hybrid3_Baseline']['group_ndcg'].max():.4f}")

print("\n2. Least Misery provides best balance")
print(f"   - Mean NDCG: {df_master[df_master['Strategy']=='Least Misery']['group_ndcg@10'].mean():.4f}")
print(f"   - Mean Fairness: {df_master[df_master['Strategy']=='Least Misery']['group_fairness'].mean():.4f}")

print("\n3. Disagreement penalty reduces NDCG but may improve fairness")
penalty_effect = penalty_data if len(penalty_data) > 0 else None
if penalty_effect is not None and len(penalty_effect) > 1:
    print(f"   - Penalty 0.0: NDCG={penalty_effect.iloc[0]['group_ndcg']:.4f}")
    print(f"   - Penalty 1.0: NDCG={penalty_effect.iloc[-1]['group_ndcg']:.4f}")

print("\n" + "="*60)
print("ALL VISUALIZATIONS CREATED SUCCESSFULLY!")
print("="*60)
