#!/usr/bin/env python3
"""
Figure 6: SNR Ablation Trade-off
Dual-axis plot showing protection vs usability across SNR ranges
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
    data_path = base_dir / 'results' / 'ablation' / 'snr_ablation.csv'
    output_path = base_dir / 'paper' / 'figures' / 'fig6_snr_ablation.pdf'
    
    # Load data
    df = pd.read_csv(data_path)
    
    # Prepare data
    snr_labels = df['snr_range'].tolist()
    protection = (df['protection'] * 100).tolist()  # Convert to percentage
    stoi = df['stoi'].tolist()
    
    # Create figure with dual y-axes
    fig, ax1 = plt.subplots(figsize=(8, 5))
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 11
    
    # X-axis positions
    x = np.arange(len(snr_labels))
    
    # Plot protection (left y-axis)
    color1 = 'black'
    ax1.set_xlabel('SNR Range (dB)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Protection (SIM Degradation %)', fontsize=12, fontweight='bold', color=color1)
    line1 = ax1.plot(x, protection, color=color1, marker='o', markersize=8, 
                     linewidth=2.5, label='Protection', linestyle='-')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim([0, 10])
    ax1.grid(axis='y', alpha=0.3, linestyle=':', zorder=0)
    
    # Create second y-axis for usability
    ax2 = ax1.twinx()
    color2 = 'gray'
    ax2.set_ylabel('Usability (STOI)', fontsize=12, fontweight='bold', color=color2)
    line2 = ax2.plot(x, stoi, color=color2, marker='s', markersize=8,
                     linewidth=2.5, label='Usability', linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim([0.93, 1.01])
    
    # Mark optimal point ([10, 20] dB)
    optimal_idx = 1
    ax1.plot(optimal_idx, protection[optimal_idx], marker='*', markersize=20,
             color='black', markeredgecolor='black', markeredgewidth=1.5, zorder=10)
    ax2.plot(optimal_idx, stoi[optimal_idx], marker='*', markersize=20,
             color='gray', markeredgecolor='gray', markeredgewidth=1.5, zorder=10)
    
    # Add annotation for optimal point
    ax1.annotate('Optimal\n[10, 20] dB', xy=(optimal_idx, protection[optimal_idx]),
                xytext=(optimal_idx + 0.5, protection[optimal_idx] + 1.5),
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', linewidth=1.5),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3', lw=1.5))
    
    # X-axis labels
    ax1.set_xticks(x)
    ax1.set_xticklabels(snr_labels, fontsize=10)
    
    # Add combined legend
    lines = line1 + line2
    labels = ['Protection (↓ SIM)', 'Usability (↑ STOI)']
    ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15),
               ncol=2, frameon=True, fontsize=10)
    
    plt.title('SNR Range Trade-off Analysis', fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    print(f"Figure saved to {output_path}")
    
    # Print summary
    print("\nSNR Ablation Summary:")
    for i, label in enumerate(snr_labels):
        print(f"{label:12s}: Protection={protection[i]:5.1f}%, STOI={stoi[i]:.3f}")
    print(f"\n✓ Optimal: [10, 20] dB balances protection ({protection[optimal_idx]:.1f}%) and usability (STOI={stoi[optimal_idx]:.3f})")

if __name__ == '__main__':
    main()

