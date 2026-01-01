
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add 'src' to path. detailed path handling.
# This script is in src/experiments/
# src is one level up.
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(src_dir)

sys.path.insert(0, src_dir)

from recommender.data_loader import load_all_data


def analyze_overlaps():
    # Define output file
    # results_dir = os.path.join(project_root, "results")
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results", "user_based_cf")
    os.makedirs(results_dir, exist_ok=True)
    output_file = os.path.join(results_dir, "overlap_analysis_results.txt")
    
    output_content = []
    
    def log(msg):
        print(msg)
        output_content.append(str(msg))

    log("Loading data...")
    # Load data
    movies, ratings, watchlists, R_cf, R_dense = load_all_data()
    
    log(f"Analyzing overlaps for {len(R_cf)} users and {len(R_cf.columns)} movies...")
    
    # Convert to boolean matrix (Present=1, Absent=0)
    R_bool = R_cf.notna().astype(int)
    
    # Compute Intersection Matrix: R_bool @ R_bool.T
    intersection_matrix = R_bool.dot(R_bool.T)
    
    # Extract upper triangle entries (excluding diagonal)
    users = R_cf.index
    vals = intersection_matrix.values
    iu1 = np.triu_indices(len(users), k=1)
    overlaps = vals[iu1]
    
    log(f"Total pairs: {len(overlaps)}")
    log(f"Min overlap: {np.min(overlaps)}")
    log(f"Max overlap: {np.max(overlaps)}")
    log(f"Mean overlap: {np.mean(overlaps):.2f}")
    log(f"Median overlap: {np.median(overlaps)}")
    
    # Calculate percentages
    thresholds = [1, 2, 3, 5, 10, 20]
    log("\nCoverage Analysis:")
    probs = []
    for t in thresholds:
        count = np.sum(overlaps >= t)
        pct = 100 * count / len(overlaps)
        log(f"Pairs with overlap >= {t}: {pct:.1f}%")
        probs.append(pct)

    # Save text results
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(output_content))
    log(f"\nResults saved to: {output_file}")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.hist(overlaps, bins=range(0, 50), edgecolor='black', alpha=0.7)
    plt.title("Distribution of Co-Rated Movies (Overlaps)")
    plt.xlabel("Number of Co-Rated Movies")
    plt.ylabel("Frequency (Pairs of Users)")
    plt.axvline(x=np.median(overlaps), color='red', linestyle='dashed', linewidth=1, label=f'Median: {np.median(overlaps)}')
    plt.legend()
    plt.grid(axis='y', alpha=0.5)
    
    plot_file = os.path.join(results_dir, "overlap_distribution.png")
    plt.savefig(plot_file)
    log(f"Plot saved to: {plot_file}")
    plt.close()

if __name__ == "__main__":
    analyze_overlaps()
