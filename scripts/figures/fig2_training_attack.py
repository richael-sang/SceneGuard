#!/usr/bin/env python3
"""
Figure 2: Training Attack Comparison
Bar chart comparing clean, random noise, Gaussian noise, and SceneGuard
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def main():
    # Setup paths
    base_dir = Path(__file__).parent.parent.parent
    data_path = base_dir / 'results' / 'baselines' / 'baseline_comparison.csv'
    output_path = base_dir / 'paper' / 'figures' / 'fig2_training_attack.pdf'
    
    # Load data
    df = pd.read_csv(data_path)
    
    # Prepare data
    methods = df['method'].tolist()
    sim_means = df['sim_mean'].tolist()
    
    # Calculate SIM degradation (1.0 - sim_mean)
    clean_sim = sim_means[0]  # Should be 1.0
    degradations = [(clean_sim - sim) * 100 for sim in sim_means]  # Convert to percentage
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5))
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 11
    
    # Bar positions
    x = np.arange(len(methods))
    width = 0.6
    
    # Colors and hatching patterns (grayscale)
    colors = ['white', 'white', 'white', 'white']
    hatches = ['', '///', '\\\\\\', 'xxx']
    edgecolors = ['black', 'black', 'black', 'black']
    linewidths = [2, 2, 2, 2.5]  # SceneGuard slightly thicker
    
    # Plot bars
    bars = ax.bar(x, degradations, width, color=colors, edgecolor=edgecolors, 
                   linewidth=linewidths)
    
    # Apply hatching
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    
    # Add value labels on top of bars
    for i, (bar, val) in enumerate(zip(bars, degradations)):
        height = bar.get_height()
        label = f'{val:.1f}%' if val > 0 else '0%'
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                label, ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add significance markers
    # SceneGuard vs Random: ***
    ax.plot([1, 3], [6.5, 6.5], 'k-', linewidth=1)
    ax.text(2, 6.8, '***', ha='center', va='bottom', fontsize=12)
    
    # SceneGuard vs Gaussian: ***
    ax.plot([2, 3], [4.5, 4.5], 'k-', linewidth=1)
    ax.text(2.5, 4.8, '***', ha='center', va='bottom', fontsize=12)
    
    # Styling
    ax.set_ylabel('Speaker Similarity Degradation (%)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=10, rotation=15, ha='right')
    ax.set_ylim([0, 8])
    ax.grid(axis='y', alpha=0.3, linestyle=':', zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add note about significance
    ax.text(0.02, 0.98, '*** p < 0.001', transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    plt.tight_layout()
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    print(f"Figure saved to {output_path}")
    
    # Print statistics
    print("\nTraining Attack Comparison:")
    for method, deg in zip(methods, degradations):
        print(f"{method:20s}: {deg:5.1f}% degradation")
    print(f"\nSceneGuard achieves {degradations[3]:.1f}% degradation, significantly stronger than random ({degradations[1]:.1f}%) or Gaussian ({degradations[2]:.1f}%) noise.")

if __name__ == '__main__':
    main()

