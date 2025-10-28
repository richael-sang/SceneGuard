#!/usr/bin/env python3
"""
Figure 4: Robustness Heatmap
Showing SIM and WER under different countermeasures
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
    data_path = base_dir / 'reports' / 'metrics' / 'robustness_summary.csv'
    output_path = base_dir / 'paper' / 'figures' / 'fig4_robustness.pdf'
    
    # Load data
    df = pd.read_csv(data_path)
    
    # Add baseline row
    baseline_row = {
        'countermeasure': 'None (baseline)',
        'sim_mean': 0.937,
        'wer_mean': 3.60
    }
    df = pd.concat([pd.DataFrame([baseline_row]), df], ignore_index=True)
    
    # Prepare data for heatmap
    countermeasures = [
        'None (baseline)',
        'MP3 128k',
        'MP3 64k',
        'Spectral Sub.',
        'Lowpass 3400Hz',
        'Downsample 8kHz'
    ]
    
    # Map from df countermeasure names to display names
    name_map = {
        'None (baseline)': 'None (baseline)',
        'mp3_128k': 'MP3 128k',
        'mp3_64k': 'MP3 64k',
        'spectral_sub_10db': 'Spectral Sub.',
        'lowpass_3400hz': 'Lowpass 3400Hz',
        'resample_8k': 'Downsample 8kHz'
    }
    
    # Create matrices
    sim_data = []
    wer_data = []
    
    for cm in countermeasures:
        # Find matching row
        for idx, row in df.iterrows():
            display_name = name_map.get(row['countermeasure'], row['countermeasure'])
            if display_name == cm:
                sim_data.append(row['sim_mean'])
                wer_data.append(row['wer_mean'])
                break
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 10
    
    # SIM heatmap
    sim_matrix = np.array(sim_data).reshape(-1, 1)
    im1 = axes[0].imshow(sim_matrix, cmap='Greys', aspect='auto', vmin=0.6, vmax=1.0)
    axes[0].set_yticks(range(len(countermeasures)))
    axes[0].set_yticklabels(countermeasures, fontsize=10)
    axes[0].set_xticks([0])
    axes[0].set_xticklabels(['Similarity'], fontsize=11)
    axes[0].set_title('(a) Speaker Similarity', fontsize=12)
    
    # Add text annotations
    for i, val in enumerate(sim_data):
        color = 'white' if val < 0.8 else 'black'
        axes[0].text(0, i, f'{val:.3f}', ha='center', va='center', 
                     color=color, fontsize=10, fontweight='bold')
    
    # Add colorbar
    cbar1 = plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    cbar1.set_label('Similarity', fontsize=10)
    
    # WER heatmap
    wer_matrix = np.array(wer_data).reshape(-1, 1)
    im2 = axes[1].imshow(wer_matrix, cmap='Greys_r', aspect='auto', vmin=0, vmax=15)
    axes[1].set_yticks(range(len(countermeasures)))
    axes[1].set_yticklabels([''] * len(countermeasures))  # Hide labels for second plot
    axes[1].set_xticks([0])
    axes[1].set_xticklabels(['WER (%)'], fontsize=11)
    axes[1].set_title('(b) Word Error Rate', fontsize=12)
    
    # Add text annotations
    for i, val in enumerate(wer_data):
        color = 'white' if val > 7.5 else 'black'
        axes[1].text(0, i, f'{val:.1f}%', ha='center', va='center', 
                     color=color, fontsize=10, fontweight='bold')
    
    # Add colorbar
    cbar2 = plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    cbar2.set_label('WER (%)', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    print(f"Figure saved to {output_path}")
    
    # Print enhancement vs degradation
    baseline_sim = sim_data[0]
    print("\nRobustness Analysis:")
    print(f"Baseline SIM: {baseline_sim:.3f}")
    for i, cm in enumerate(countermeasures[1:], 1):
        delta = sim_data[i] - baseline_sim
        status = "Enhanced" if delta < -0.05 else "Maintained"
        print(f"{cm:20s}: SIM={sim_data[i]:.3f}, Δ={delta:+.3f} ({status})")

if __name__ == '__main__':
    main()

