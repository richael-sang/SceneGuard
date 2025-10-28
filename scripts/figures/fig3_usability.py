#!/usr/bin/env python3
"""
Figure 3: Usability Metrics Distribution
Box plots showing STOI, WER, and PESQ distributions
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

def main():
    # Setup paths
    base_dir = Path(__file__).parent.parent.parent
    data_path = base_dir / 'reports' / 'metrics' / 'defense_evaluation.csv'
    output_path = base_dir / 'paper' / 'figures' / 'fig3_usability.pdf'
    
    # Load data
    df = pd.read_csv(data_path)
    
    # Prepare data
    stoi_values = df['stoi'].values
    wer_values = df['wer'].values * 100  # Convert to percentage
    pesq_values = df['pesq'].values
    
    # Set up figure with academic style (grayscale, patterns)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 10
    
    # STOI
    bp1 = axes[0].boxplot([stoi_values], widths=0.6, patch_artist=True,
                           boxprops=dict(facecolor='white', edgecolor='black', linewidth=1.5),
                           medianprops=dict(color='black', linewidth=2),
                           whiskerprops=dict(color='black', linewidth=1.5),
                           capprops=dict(color='black', linewidth=1.5),
                           flierprops=dict(marker='o', markerfacecolor='white', 
                                          markeredgecolor='black', markersize=6))
    axes[0].axhline(y=0.85, color='gray', linestyle='--', linewidth=1.5, label='Threshold (0.85)')
    axes[0].set_ylabel('STOI', fontsize=12)
    axes[0].set_ylim([0.75, 1.05])
    axes[0].set_xticks([])
    axes[0].grid(axis='y', alpha=0.3, linestyle=':')
    axes[0].legend(loc='lower right', fontsize=9)
    axes[0].set_title('(a) Intelligibility', fontsize=12)
    
    # WER
    bp2 = axes[1].boxplot([wer_values], widths=0.6, patch_artist=True,
                           boxprops=dict(facecolor='white', edgecolor='black', linewidth=1.5, hatch='///'),
                           medianprops=dict(color='black', linewidth=2),
                           whiskerprops=dict(color='black', linewidth=1.5),
                           capprops=dict(color='black', linewidth=1.5),
                           flierprops=dict(marker='o', markerfacecolor='white', 
                                          markeredgecolor='black', markersize=6))
    axes[1].axhline(y=15, color='gray', linestyle='--', linewidth=1.5, label='Threshold (15%)')
    axes[1].set_ylabel('WER (%)', fontsize=12)
    axes[1].set_ylim([-5, 25])
    axes[1].set_xticks([])
    axes[1].grid(axis='y', alpha=0.3, linestyle=':')
    axes[1].legend(loc='upper right', fontsize=9)
    axes[1].set_title('(b) Transcription Accuracy', fontsize=12)
    
    # PESQ
    bp3 = axes[2].boxplot([pesq_values], widths=0.6, patch_artist=True,
                           boxprops=dict(facecolor='white', edgecolor='black', linewidth=1.5, hatch='\\\\\\'),
                           medianprops=dict(color='black', linewidth=2),
                           whiskerprops=dict(color='black', linewidth=1.5),
                           capprops=dict(color='black', linewidth=1.5),
                           flierprops=dict(marker='o', markerfacecolor='white', 
                                          markeredgecolor='black', markersize=6))
    axes[2].axhline(y=3.0, color='gray', linestyle='--', linewidth=1.5, label='Ideal (3.0)')
    axes[2].set_ylabel('PESQ', fontsize=12)
    axes[2].set_ylim([0.5, 4.5])
    axes[2].set_xticks([])
    axes[2].grid(axis='y', alpha=0.3, linestyle=':')
    axes[2].legend(loc='lower right', fontsize=9)
    axes[2].set_title('(c) Perceptual Quality', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    print(f"Figure saved to {output_path}")
    
    # Print statistics
    print("\nUsability Statistics:")
    print(f"STOI:  mean={np.mean(stoi_values):.4f}, std={np.std(stoi_values):.4f}")
    print(f"WER:   mean={np.mean(wer_values):.2f}%, std={np.std(wer_values):.2f}%")
    print(f"PESQ:  mean={np.mean(pesq_values):.3f}, std={np.std(pesq_values):.3f}")

if __name__ == '__main__':
    main()

