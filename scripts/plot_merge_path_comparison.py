#!/usr/bin/env python3
"""
Plot performance comparison between merge_path and block_mapped load balancing methods.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import sys

# Try to import seaborn, but continue without it if not available
try:
    import seaborn as sns
    sns.set_style("whitegrid")
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

# Set style for publication-ready plots
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

def load_data(csv_file):
    """Load benchmark data from CSV file."""
    try:
        df = pd.read_csv(csv_file)
        print(f"Loaded {len(df)} rows from {csv_file}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}", file=sys.stderr)
        sys.exit(1)

def calculate_statistics(df):
    """Calculate summary statistics for each configuration."""
    stats = df.groupby(['dataset', 'algorithm', 'load_balance']).agg({
        'time_ms': ['mean', 'median', 'std', 'min', 'max', 'count']
    }).reset_index()
    stats.columns = ['dataset', 'algorithm', 'load_balance', 'mean', 'median', 'std', 'min', 'max', 'count']
    return stats

def plot_bar_comparison(df, output_dir):
    """Create bar chart comparing average execution times."""
    stats = calculate_statistics(df)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for idx, algo in enumerate(['sssp', 'bfs']):
        ax = axes[idx]
        algo_data = stats[stats['algorithm'] == algo]
        
        x = np.arange(len(algo_data['dataset'].unique()))
        width = 0.35
        datasets = sorted(algo_data['dataset'].unique())
        
        merge_path_means = []
        block_mapped_means = []
        merge_path_stds = []
        block_mapped_stds = []
        
        for dataset in datasets:
            mp_data = algo_data[(algo_data['dataset'] == dataset) & 
                               (algo_data['load_balance'] == 'merge_path')]
            bm_data = algo_data[(algo_data['dataset'] == dataset) & 
                               (algo_data['load_balance'] == 'block_mapped')]
            
            # Only add data if both methods have results
            if len(mp_data) > 0 and len(bm_data) > 0:
                merge_path_means.append(mp_data['mean'].values[0])
                block_mapped_means.append(bm_data['mean'].values[0])
                merge_path_stds.append(mp_data['std'].values[0] if not mp_data['std'].isna().values[0] else 0)
                block_mapped_stds.append(bm_data['std'].values[0] if not bm_data['std'].isna().values[0] else 0)
            elif len(mp_data) > 0:
                merge_path_means.append(mp_data['mean'].values[0])
                block_mapped_means.append(0)  # Missing data
                merge_path_stds.append(mp_data['std'].values[0] if not mp_data['std'].isna().values[0] else 0)
                block_mapped_stds.append(0)
            elif len(bm_data) > 0:
                merge_path_means.append(0)  # Missing data
                block_mapped_means.append(bm_data['mean'].values[0])
                merge_path_stds.append(0)
                block_mapped_stds.append(bm_data['std'].values[0] if not bm_data['std'].isna().values[0] else 0)
            else:
                # Skip datasets with no data
                continue
        
        bars1 = ax.bar(x - width/2, merge_path_means, width, 
                      label='merge_path', yerr=merge_path_stds, capsize=5, alpha=0.8)
        bars2 = ax.bar(x + width/2, block_mapped_means, width,
                      label='block_mapped', yerr=block_mapped_stds, capsize=5, alpha=0.8)
        
        ax.set_xlabel('Dataset', fontweight='bold')
        ax.set_ylabel('Execution Time (ms)', fontweight='bold')
        ax.set_title(f'{algo.upper()} Performance Comparison', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}',
                       ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'merge_path_vs_block_mapped_bar.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved bar chart to {output_file}")
    plt.close()

def plot_boxplot_comparison(df, output_dir):
    """Create box plot showing distribution of execution times."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for idx, algo in enumerate(['sssp', 'bfs']):
        ax = axes[idx]
        algo_data = df[df['algorithm'] == algo]
        
        # Prepare data for boxplot
        plot_data = []
        labels = []
        positions = []
        pos = 0
        
        datasets = sorted(algo_data['dataset'].unique())
        for dataset in datasets:
            for lb in ['merge_path', 'block_mapped']:
                data = algo_data[(algo_data['dataset'] == dataset) & 
                                (algo_data['load_balance'] == lb)]['time_ms'].values
                if len(data) > 0:
                    plot_data.append(data)
                    labels.append(f"{dataset}\n{lb}")
                    positions.append(pos)
                    pos += 1
            pos += 0.5  # Add spacing between datasets
        
        if plot_data:
            bp = ax.boxplot(plot_data, positions=positions, widths=0.6, patch_artist=True)
            
            # Color the boxes
            colors = ['lightblue', 'lightcoral']
            for i, patch in enumerate(bp['boxes']):
                patch.set_facecolor(colors[i % 2])
                patch.set_alpha(0.7)
        
        ax.set_xlabel('Dataset and Load Balance Method', fontweight='bold')
        ax.set_ylabel('Execution Time (ms)', fontweight='bold')
        ax.set_title(f'{algo.upper()} Execution Time Distribution', fontweight='bold')
        ax.set_xticks(positions[::2])  # Show only dataset names
        ax.set_xticklabels(datasets, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='lightblue', alpha=0.7, label='merge_path'),
                          Patch(facecolor='lightcoral', alpha=0.7, label='block_mapped')]
        ax.legend(handles=legend_elements)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'merge_path_vs_block_mapped_boxplot.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved box plot to {output_file}")
    plt.close()

def plot_speedup_comparison(df, output_dir):
    """Create speedup plot showing relative performance."""
    stats = calculate_statistics(df)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for idx, algo in enumerate(['sssp', 'bfs']):
        ax = axes[idx]
        algo_data = stats[stats['algorithm'] == algo]
        
        datasets = sorted(algo_data['dataset'].unique())
        speedups = []
        speedup_stds = []
        
        for dataset in datasets:
            mp_data = algo_data[(algo_data['dataset'] == dataset) & 
                               (algo_data['load_balance'] == 'merge_path')]
            bm_data = algo_data[(algo_data['dataset'] == dataset) & 
                               (algo_data['load_balance'] == 'block_mapped')]
            
            if len(mp_data) > 0 and len(bm_data) > 0:
                # Speedup = block_mapped_time / merge_path_time
                # > 1 means merge_path is faster
                speedup = bm_data['mean'].values[0] / mp_data['mean'].values[0]
                
                # Approximate std using error propagation
                mp_mean = mp_data['mean'].values[0]
                bm_mean = bm_data['mean'].values[0]
                mp_std = mp_data['std'].values[0]
                bm_std = bm_data['std'].values[0]
                speedup_std = speedup * np.sqrt((bm_std/bm_mean)**2 + (mp_std/mp_mean)**2)
                
                speedups.append(speedup)
                speedup_stds.append(speedup_std)
            else:
                speedups.append(1.0)
                speedup_stds.append(0.0)
        
        x = np.arange(len(datasets))
        bars = ax.bar(x, speedups, yerr=speedup_stds, capsize=5, alpha=0.8, 
                     color=['green' if s > 1 else 'red' for s in speedups])
        
        # Add horizontal line at y=1
        ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        
        # Add value labels
        for i, (bar, speedup) in enumerate(zip(bars, speedups)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{speedup:.2f}x',
                   ha='center', va='bottom' if speedup > 1 else 'top', fontsize=9, fontweight='bold')
        
        ax.set_xlabel('Dataset', fontweight='bold')
        ax.set_ylabel('Speedup (block_mapped / merge_path)', fontweight='bold')
        ax.set_title(f'{algo.upper()} Speedup Comparison\n(>1 means merge_path is faster)', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'merge_path_vs_block_mapped_speedup.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved speedup plot to {output_file}")
    plt.close()

def plot_combined_summary(df, output_dir):
    """Create a combined summary plot with all comparisons."""
    stats = calculate_statistics(df)
    
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Bar chart for SSSP
    ax1 = fig.add_subplot(gs[0, 0])
    algo = 'sssp'
    algo_data = stats[stats['algorithm'] == algo]
    datasets = sorted(algo_data['dataset'].unique())
    x = np.arange(len(datasets))
    width = 0.35
    
    merge_path_means = [algo_data[(algo_data['dataset'] == d) & 
                                  (algo_data['load_balance'] == 'merge_path')]['mean'].values[0] 
                        if len(algo_data[(algo_data['dataset'] == d) & 
                                        (algo_data['load_balance'] == 'merge_path')]) > 0 else 0
                        for d in datasets]
    block_mapped_means = [algo_data[(algo_data['dataset'] == d) & 
                                    (algo_data['load_balance'] == 'block_mapped')]['mean'].values[0] 
                          if len(algo_data[(algo_data['dataset'] == d) & 
                                          (algo_data['load_balance'] == 'block_mapped')]) > 0 else 0
                          for d in datasets]
    
    ax1.bar(x - width/2, merge_path_means, width, label='merge_path', alpha=0.8)
    ax1.bar(x + width/2, block_mapped_means, width, label='block_mapped', alpha=0.8)
    ax1.set_xlabel('Dataset')
    ax1.set_ylabel('Time (ms)')
    ax1.set_title(f'{algo.upper()} Average Execution Time')
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Bar chart for BFS
    ax2 = fig.add_subplot(gs[0, 1])
    algo = 'bfs'
    algo_data = stats[stats['algorithm'] == algo]
    datasets = sorted(algo_data['dataset'].unique())
    x = np.arange(len(datasets))
    
    merge_path_means = [algo_data[(algo_data['dataset'] == d) & 
                                  (algo_data['load_balance'] == 'merge_path')]['mean'].values[0] 
                        if len(algo_data[(algo_data['dataset'] == d) & 
                                        (algo_data['load_balance'] == 'merge_path')]) > 0 else 0
                        for d in datasets]
    block_mapped_means = [algo_data[(algo_data['dataset'] == d) & 
                                    (algo_data['load_balance'] == 'block_mapped')]['mean'].values[0] 
                          if len(algo_data[(algo_data['dataset'] == d) & 
                                          (algo_data['load_balance'] == 'block_mapped')]) > 0 else 0
                          for d in datasets]
    
    ax2.bar(x - width/2, merge_path_means, width, label='merge_path', alpha=0.8)
    ax2.bar(x + width/2, block_mapped_means, width, label='block_mapped', alpha=0.8)
    ax2.set_xlabel('Dataset')
    ax2.set_ylabel('Time (ms)')
    ax2.set_title(f'{algo.upper()} Average Execution Time')
    ax2.set_xticks(x)
    ax2.set_xticklabels(datasets, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # Speedup for SSSP
    ax3 = fig.add_subplot(gs[1, 0])
    algo = 'sssp'
    algo_data = stats[stats['algorithm'] == algo]
    datasets = sorted(algo_data['dataset'].unique())
    speedups = []
    for dataset in datasets:
        mp_data = algo_data[(algo_data['dataset'] == dataset) & 
                           (algo_data['load_balance'] == 'merge_path')]
        bm_data = algo_data[(algo_data['dataset'] == dataset) & 
                           (algo_data['load_balance'] == 'block_mapped')]
        if len(mp_data) > 0 and len(bm_data) > 0:
            speedups.append(bm_data['mean'].values[0] / mp_data['mean'].values[0])
        else:
            speedups.append(1.0)
    
    x = np.arange(len(datasets))
    bars = ax3.bar(x, speedups, alpha=0.8, color=['green' if s > 1 else 'red' for s in speedups])
    ax3.axhline(y=1.0, color='black', linestyle='--', linewidth=1)
    ax3.set_xlabel('Dataset')
    ax3.set_ylabel('Speedup')
    ax3.set_title(f'{algo.upper()} Speedup (block_mapped/merge_path)')
    ax3.set_xticks(x)
    ax3.set_xticklabels(datasets, rotation=45, ha='right')
    ax3.grid(axis='y', alpha=0.3)
    
    # Speedup for BFS
    ax4 = fig.add_subplot(gs[1, 1])
    algo = 'bfs'
    algo_data = stats[stats['algorithm'] == algo]
    datasets = sorted(algo_data['dataset'].unique())
    speedups = []
    for dataset in datasets:
        mp_data = algo_data[(algo_data['dataset'] == dataset) & 
                           (algo_data['load_balance'] == 'merge_path')]
        bm_data = algo_data[(algo_data['dataset'] == dataset) & 
                           (algo_data['load_balance'] == 'block_mapped')]
        if len(mp_data) > 0 and len(bm_data) > 0:
            speedups.append(bm_data['mean'].values[0] / mp_data['mean'].values[0])
        else:
            speedups.append(1.0)
    
    x = np.arange(len(datasets))
    bars = ax4.bar(x, speedups, alpha=0.8, color=['green' if s > 1 else 'red' for s in speedups])
    ax4.axhline(y=1.0, color='black', linestyle='--', linewidth=1)
    ax4.set_xlabel('Dataset')
    ax4.set_ylabel('Speedup')
    ax4.set_title(f'{algo.upper()} Speedup (block_mapped/merge_path)')
    ax4.set_xticks(x)
    ax4.set_xticklabels(datasets, rotation=45, ha='right')
    ax4.grid(axis='y', alpha=0.3)
    
    output_file = os.path.join(output_dir, 'merge_path_vs_block_mapped_summary.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved summary plot to {output_file}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Plot merge_path vs block_mapped performance comparison')
    parser.add_argument('csv_file', help='Input CSV file with benchmark results')
    parser.add_argument('--output-dir', '-o', default='benchmarks/plots',
                       help='Output directory for plots (default: benchmarks/plots)')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    df = load_data(args.csv_file)
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("=" * 80)
    stats = calculate_statistics(df)
    print(stats.to_string())
    print("=" * 80)
    
    # Generate all plots
    print("\nGenerating plots...")
    plot_bar_comparison(df, args.output_dir)
    plot_boxplot_comparison(df, args.output_dir)
    plot_speedup_comparison(df, args.output_dir)
    plot_combined_summary(df, args.output_dir)
    
    print("\nAll plots generated successfully!")

if __name__ == '__main__':
    main()
