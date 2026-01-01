"""
UBCF Grid Search Results Visualization
Creates comprehensive visualizations for UBCF hyperparameter optimization
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Read the data
results_dir = Path(__file__).parent / "results"
df = pd.read_csv(results_dir / "ubcf_grid_search_ndcg.csv")

# Clean data
df = df.dropna()

# Get unique values
methods = df['Method'].unique()
k_values = sorted(df['K_Neighbors'].unique())
overlap_values = sorted(df['Min_Overlap'].unique())

# Define colors for methods
method_colors = {
    'Cosine': '#e74c3c',
    'Pearson_SW': '#3498db',
    'Pearson_Shrink': '#2ecc71',
    'Spearman_Rank': '#f39c12',
    'Spearman_SW': '#9b59b6'
}

method_markers = {
    'Cosine': 'o',
    'Pearson_SW': 's',
    'Pearson_Shrink': '^',
    'Spearman_Rank': 'D',
    'Spearman_SW': 'v'
}

def create_visualization(theme='dark'):
    """Create all visualizations for a given theme"""
    
    # Theme settings
    if theme == 'dark':
        bg_color = '#151515'
        text_color = 'white'
        grid_color = 'white'
        legend_bg = '#1a1a1a'
        suffix = ''
    else:  # white theme
        bg_color = 'white'
        text_color = 'black'
        grid_color = 'gray'
        legend_bg = 'white'
        suffix = '_white'
    
    # ============================================================================
    # Figure 1: Impact of K on NDCG@10 (grouped by Min_Overlap)
    # ============================================================================
    fig1, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig1.patch.set_facecolor(bg_color)
    
    for idx, overlap in enumerate(overlap_values):
        ax = axes[idx]
        ax.set_facecolor(bg_color)
        
        subset = df[df['Min_Overlap'] == overlap]
        
        for method in methods:
            method_data = subset[subset['Method'] == method]
            if len(method_data) > 0:
                ax.plot(method_data['K_Neighbors'], 
                       method_data['NDCG@10'],
                       marker=method_markers[method],
                       color=method_colors[method],
                       label=method,
                       linewidth=2.5,
                       markersize=10,
                       alpha=0.85)
        
        ax.set_xlabel('K (Number of Neighbors)', fontsize=12, fontweight='bold', color=text_color)
        ax.set_ylabel('NDCG@10', fontsize=12, fontweight='bold', color=text_color)
        ax.set_title(f'Min Overlap = {int(overlap)}', fontsize=13, fontweight='bold', color=text_color)
        ax.grid(True, alpha=0.3, linestyle='--', color=grid_color)
        ax.tick_params(axis='both', colors=text_color, labelsize=10)
        
        # Set spines color
        for spine in ax.spines.values():
            spine.set_color(text_color)
        
        if idx == 1:
            ax.legend(fontsize=10, loc='best', facecolor=legend_bg, 
                     edgecolor=text_color, framealpha=0.9, labelcolor=text_color)
    
    fig1.suptitle('UBCF: Impact of K on NDCG@10 Performance', 
                  fontsize=16, fontweight='bold', color=text_color, y=0.98)
    plt.tight_layout()
    plt.savefig(results_dir / f'ubcf_k_impact{suffix}.svg', format='svg', dpi=300, facecolor=bg_color)
    print(f"✓ Figure 1 ({theme} theme) saved: ubcf_k_impact{suffix}.svg")
    plt.close()
    
    # ============================================================================
    # Figure 2: Method Comparison Heatmap
    # ============================================================================
    fig2, ax = plt.subplots(figsize=(10, 6))
    fig2.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    
    # Create pivot table for heatmap
    pivot_data = df.pivot_table(values='NDCG@10', 
                                 index='Method', 
                                 columns=['K_Neighbors', 'Min_Overlap'],
                                 aggfunc='mean')
    
    # Create heatmap
    im = ax.imshow(pivot_data.values, cmap='YlOrRd', aspect='auto', vmin=0.2, vmax=0.42)
    
    # Set ticks
    ax.set_xticks(np.arange(len(pivot_data.columns)))
    ax.set_yticks(np.arange(len(pivot_data.index)))
    
    # Create labels
    col_labels = [f'K={k}\nO={o}' for k, o in pivot_data.columns]
    ax.set_xticklabels(col_labels, color=text_color)
    ax.set_yticklabels(pivot_data.index, color=text_color)
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('NDCG@10', rotation=270, labelpad=20, fontsize=12, 
                   fontweight='bold', color=text_color)
    cbar.ax.yaxis.set_tick_params(color=text_color)
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=text_color)
    
    # Add text annotations
    for i in range(len(pivot_data.index)):
        for j in range(len(pivot_data.columns)):
            value = pivot_data.values[i, j]
            if not np.isnan(value):
                text = ax.text(j, i, f'{value:.3f}',
                             ha="center", va="center", color="black", 
                             fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Configuration (K, Min Overlap)', fontsize=12, fontweight='bold', color=text_color)
    ax.set_ylabel('Similarity Method', fontsize=12, fontweight='bold', color=text_color)
    ax.set_title('UBCF: Performance Heatmap Across All Configurations', 
                 fontsize=14, fontweight='bold', pad=20, color=text_color)
    ax.tick_params(axis='both', colors=text_color)
    
    plt.tight_layout()
    plt.savefig(results_dir / f'ubcf_heatmap{suffix}.svg', format='svg', dpi=300, facecolor=bg_color)
    print(f"✓ Figure 2 ({theme} theme) saved: ubcf_heatmap{suffix}.svg")
    plt.close()
    
    # ============================================================================
    # Figure 3: Bar Chart - Best Configuration per Method
    # ============================================================================
    fig3, ax = plt.subplots(figsize=(12, 7))
    fig3.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    
    # Find best configuration for each method
    best_configs = []
    for method in methods:
        method_data = df[df['Method'] == method]
        best_idx = method_data['NDCG@10'].idxmax()
        best_row = method_data.loc[best_idx]
        best_configs.append({
            'Method': method,
            'NDCG@10': best_row['NDCG@10'],
            'K': int(best_row['K_Neighbors']),
            'Overlap': int(best_row['Min_Overlap'])
        })
    
    best_df = pd.DataFrame(best_configs).sort_values('NDCG@10', ascending=False)
    
    # Create bars
    x_pos = np.arange(len(best_df))
    bars = ax.bar(x_pos, best_df['NDCG@10'], 
                  color=[method_colors[m] for m in best_df['Method']],
                  alpha=0.85, edgecolor=text_color, linewidth=1.5)
    
    # Add value labels on bars
    for i, (bar, row) in enumerate(zip(bars, best_df.itertuples())):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}\nK={row.K}, O={row.Overlap}',
                ha='center', va='bottom', fontsize=9, fontweight='bold', color=text_color)
    
    ax.set_xlabel('Similarity Method', fontsize=12, fontweight='bold', color=text_color)
    ax.set_ylabel('Best NDCG@10', fontsize=12, fontweight='bold', color=text_color)
    ax.set_title('UBCF: Best Performance per Similarity Method', 
                 fontsize=14, fontweight='bold', color=text_color)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(best_df['Method'], rotation=15, ha='right', color=text_color)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y', color=grid_color)
    ax.tick_params(axis='both', colors=text_color)
    ax.set_ylim(0, 0.45)
    
    # Set spines color
    for spine in ax.spines.values():
        spine.set_color(text_color)
    
    plt.tight_layout()
    plt.savefig(results_dir / f'ubcf_best_configs{suffix}.svg', format='svg', dpi=300, facecolor=bg_color)
    print(f"✓ Figure 3 ({theme} theme) saved: ubcf_best_configs{suffix}.svg")
    plt.close()
    
    # ============================================================================
    # Figure 4: Impact of Min_Overlap (grouped by K)
    # ============================================================================
    fig4, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig4.patch.set_facecolor(bg_color)
    
    for idx, k in enumerate(k_values):
        ax = axes[idx]
        ax.set_facecolor(bg_color)
        
        subset = df[df['K_Neighbors'] == k]
        
        for method in methods:
            method_data = subset[subset['Method'] == method]
            if len(method_data) > 0:
                ax.plot(method_data['Min_Overlap'], 
                       method_data['NDCG@10'],
                       marker=method_markers[method],
                       color=method_colors[method],
                       label=method,
                       linewidth=2.5,
                       markersize=10,
                       alpha=0.85)
        
        ax.set_xlabel('Min Overlap', fontsize=12, fontweight='bold', color=text_color)
        ax.set_ylabel('NDCG@10', fontsize=12, fontweight='bold', color=text_color)
        ax.set_title(f'K = {int(k)} Neighbors', fontsize=13, fontweight='bold', color=text_color)
        ax.grid(True, alpha=0.3, linestyle='--', color=grid_color)
        ax.tick_params(axis='both', colors=text_color, labelsize=10)
        
        # Set spines color
        for spine in ax.spines.values():
            spine.set_color(text_color)
        
        if idx == 1:
            ax.legend(fontsize=10, loc='best', facecolor=legend_bg, 
                     edgecolor=text_color, framealpha=0.9, labelcolor=text_color)
    
    fig4.suptitle('UBCF: Impact of Min Overlap on NDCG@10 Performance', 
                  fontsize=16, fontweight='bold', color=text_color, y=0.98)
    plt.tight_layout()
    plt.savefig(results_dir / f'ubcf_overlap_impact{suffix}.svg', format='svg', dpi=300, facecolor=bg_color)
    print(f"✓ Figure 4 ({theme} theme) saved: ubcf_overlap_impact{suffix}.svg")
    plt.close()

# Create both themes
print("\n" + "="*70)
print("UBCF GRID SEARCH VISUALIZATION")
print("="*70)

print("\n🌙 Creating DARK theme visualizations...")
create_visualization(theme='dark')

print("\n☀️  Creating WHITE theme visualizations...")
create_visualization(theme='white')

# Print summary
print("\n" + "="*70)
print("✅ ALL VISUALIZATIONS CREATED SUCCESSFULLY!")
print("="*70)
print("\n📊 Generated Files:")
print("   Dark Theme (#151515):")
print("      - ubcf_k_impact.svg")
print("      - ubcf_heatmap.svg")
print("      - ubcf_best_configs.svg")
print("      - ubcf_overlap_impact.svg")
print("\n   White Theme:")
print("      - ubcf_k_impact_white.svg")
print("      - ubcf_heatmap_white.svg")
print("      - ubcf_best_configs_white.svg")
print("      - ubcf_overlap_impact_white.svg")

# Print best configuration
best_overall = df.loc[df['NDCG@10'].idxmax()]
print(f"\n🏆 BEST OVERALL CONFIGURATION:")
print(f"   Method: {best_overall['Method']}")
print(f"   K Neighbors: {int(best_overall['K_Neighbors'])}")
print(f"   Min Overlap: {int(best_overall['Min_Overlap'])}")
print(f"   NDCG@10: {best_overall['NDCG@10']:.6f}")
print("\n" + "="*70)
