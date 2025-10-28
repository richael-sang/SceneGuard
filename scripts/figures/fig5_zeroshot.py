#!/usr/bin/env python3
"""
Figure 5: Zero-Shot Attack Success Rate
Grouped bar chart comparing clean vs defended reference
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
    data_path = base_dir / 'reports' / 'metrics' / 'zeroshot_summary.csv'
    output_path = base_dir / 'paper' / 'figures' / 'fig5_zeroshot.pdf'
    
    # Load data
    df = pd.read_csv(data_path)
    
    # Extract data
    clean_success = df[df['Reference Type'] == 'Clean']['Attack Success Rate (%)'].values[0]
    defended_success = df[df['Reference Type'] == 'Defended']['Attack Success Rate (%)'].values[0]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(6, 5))
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 11
    
    # Bar positions
    x = np.array([0, 1])
    width = 0.6
    
    # Plot bars
    bars = ax.bar(x, [clean_success, defended_success], width, 
                   color='white', edgecolor='black', linewidth=2)
    
    # Add hatching for distinction
    bars[0].set_hatch(None)
    bars[1].set_hatch('///')
    
    # Add value labels on top of bars
    ax.text(0, clean_success + 1, f'{clean_success:.1f}%', 
            ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.text(1, defended_success + 1, f'{defended_success:.1f}%', 
            ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Add reduction annotation
    reduction_pct = (clean_success - defended_success) / clean_success * 100
    mid_y = (clean_success + defended_success) / 2
    ax.annotate('', xy=(1, clean_success), xytext=(1, defended_success),
                arrowprops=dict(arrowstyle='<->', color='gray', lw=1.5))
    ax.text(1.15, mid_y, f'{reduction_pct:.1f}%\nreduction', 
            va='center', fontsize=10, color='gray')
    
    # Styling
    ax.set_ylabel('Attack Success Rate (%)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(['Clean Reference', 'Defended Reference'], fontsize=11)
    ax.set_ylim([0, 30])
    ax.grid(axis='y', alpha=0.3, linestyle=':')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    print(f"Figure saved to {output_path}")
    
    # Print statistics
    print("\nZero-Shot Attack Statistics:")
    print(f"Clean Reference:    {clean_success:.1f}% success rate")
    print(f"Defended Reference: {defended_success:.1f}% success rate")
    print(f"Reduction:          {clean_success - defended_success:.1f} percentage points ({reduction_pct:.1f}% relative)")

if __name__ == '__main__':
    main()

