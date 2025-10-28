#!/usr/bin/env python3
"""
重新生成Ablation章节的figures
使用现代机器学习配色方案
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 设置现代ML配色方案
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.3)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['legend.fontsize'] = 10

# 使用seaborn的颜色方案
COLORS = {
    'primary': '#3498db',  # 蓝色
    'secondary': '#e74c3c',  # 红色
    'success': '#2ecc71',  # 绿色
    'warning': '#f39c12',  # 橙色
}


def fig6_snr_ablation():
    """Figure 6: SNR Range Ablation"""
    print("\n[1/1] Generating Figure 6: SNR Ablation...")
    
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
    
    # Data
    snr_ranges = ['[5, 10]', '[10, 20]', '[15, 25]', '[20, 30]']
    snr_centers = [7.5, 15, 20, 25]
    
    # Protection (Speaker Similarity)
    sim_values = [0.921, 0.945, 0.968, 0.982]
    protection_pct = [(1.0 - s) * 100 for s in sim_values]  # Convert to protection %
    
    # Usability (STOI)
    stoi_values = [0.942, 0.986, 0.993, 0.997]
    
    # Create twin axis
    ax2 = ax1.twinx()
    
    # Plot Protection (left axis)
    line1 = ax1.plot(snr_centers, protection_pct, 'o-', linewidth=3, markersize=10,
                     color=COLORS['primary'], label='Protection (%)', zorder=3)
    ax1.fill_between(snr_centers, protection_pct, alpha=0.2, color=COLORS['primary'])
    
    # Plot Usability (right axis)
    line2 = ax2.plot(snr_centers, stoi_values, 's-', linewidth=3, markersize=10,
                     color=COLORS['success'], label='Usability (STOI)', zorder=3)
    ax2.fill_between(snr_centers, stoi_values, alpha=0.2, color=COLORS['success'])
    
    # Highlight optimal choice
    optimal_idx = 1  # [10, 20] dB
    ax1.scatter([snr_centers[optimal_idx]], [protection_pct[optimal_idx]], 
               s=300, marker='*', color='gold', edgecolors='black', linewidths=2, 
               zorder=4, label='Optimal (Default)')
    
    # Labels and title
    ax1.set_xlabel('SNR Range Center (dB)', fontweight='bold', fontsize=13)
    ax1.set_ylabel('Protection Strength (%)', color=COLORS['primary'], 
                  fontweight='bold', fontsize=13)
    ax2.set_ylabel('Usability (STOI)', color=COLORS['success'], 
                  fontweight='bold', fontsize=13)
    ax1.set_title('SNR Range Ablation: Protection vs Usability Trade-off', 
                 fontweight='bold', fontsize=14)
    
    # Customize ticks
    ax1.set_xticks(snr_centers)
    ax1.set_xticklabels(snr_ranges, fontsize=11)
    ax1.tick_params(axis='y', labelcolor=COLORS['primary'])
    ax2.tick_params(axis='y', labelcolor=COLORS['success'])
    
    # Set y-axis limits
    ax1.set_ylim([1, 9])
    ax2.set_ylim([0.93, 1.0])
    
    # Grid
    ax1.grid(True, alpha=0.3, which='both')
    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    labels.append('Optimal (Default)')
    ax1.legend(lines + [plt.Line2D([0], [0], marker='*', color='w', 
                                   markerfacecolor='gold', markersize=15, 
                                   markeredgecolor='black', markeredgewidth=2)],
              labels, loc='center left', fontsize=11, framealpha=0.9)
    
    # Add text annotations
    for i, (center, prot, stoi) in enumerate(zip(snr_centers, protection_pct, stoi_values)):
        ax1.text(center, prot + 0.3, f'{prot:.1f}%', ha='center', va='bottom',
                fontweight='bold', fontsize=9, color=COLORS['primary'])
        ax2.text(center, stoi + 0.002, f'{stoi:.3f}', ha='center', va='bottom',
                fontweight='bold', fontsize=9, color=COLORS['success'])
    
    plt.tight_layout()
    
    output_path = Path("paper/figures/fig6_snr_ablation.pdf")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {output_path}")


def fig7_optimization_comparison():
    """Figure 7: Direct vs Optimized Comparison"""
    print("\n[2/2] Generating Figure 7: Optimization Comparison...")
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    methods = ['Direct\nMixing', 'Optimized\n(Ours)']
    
    # Data
    snr_mean = [15.0, 18.51]
    snr_std = [2.9, 0.04]
    
    sim_mean = [None, -0.378]  # Direct mixing没有similarity数据
    sim_std = [None, 0.110]
    
    stoi_mean = [0.989, 0.986]
    stoi_std = [0.005, 0.003]
    
    colors = [COLORS['warning'], COLORS['primary']]
    
    # (a) SNR Consistency
    bars1 = axes[0].bar(methods, snr_mean, yerr=snr_std, capsize=8,
                       color=colors, edgecolor='black', linewidth=1.5, alpha=0.85)
    axes[0].set_ylabel('SNR (dB)', fontweight='bold')
    axes[0].set_title('(a) SNR Consistency', fontweight='bold')
    axes[0].axhline(y=10, color='red', linestyle='--', linewidth=1.5, alpha=0.4)
    axes[0].axhline(y=20, color='red', linestyle='--', linewidth=1.5, alpha=0.4)
    axes[0].fill_between([-0.5, 1.5], 10, 20, alpha=0.1, color='gray', label='Target Range')
    axes[0].set_ylim([10, 22])
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].legend(loc='upper right')
    
    # 添加数值标签
    for bar, mean, std in zip(bars1, snr_mean, snr_std):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height + std + 0.3,
                    f'{mean:.2f}\n±{std:.2f}', ha='center', va='bottom',
                    fontweight='bold', fontsize=10)
    
    # 标注方差减少
    axes[0].text(0.5, 21, 'σ reduced by 98.6%', ha='center', va='top',
                fontsize=11, fontweight='bold', color='red',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    # (b) Speaker Similarity (Only Optimized)
    # 只显示Optimized的数据
    bars2 = axes[1].bar([methods[1]], [sim_mean[1]], yerr=[sim_std[1]], capsize=8,
                       color=[colors[1]], edgecolor='black', linewidth=1.5, alpha=0.85)
    axes[1].set_ylabel('Speaker Similarity', fontweight='bold')
    axes[1].set_title('(b) Protection Strength', fontweight='bold')
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.5)
    axes[1].axhline(y=0.25, color='orange', linestyle='--', linewidth=1.5, alpha=0.6, label='Target')
    axes[1].set_ylim([-0.6, 0.4])
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].set_xlim([-0.5, 0.5])
    axes[1].legend()
    
    # 添加数值标签
    axes[1].text(0, sim_mean[1] - sim_std[1] - 0.05,
                f'{sim_mean[1]:.3f}\n±{sim_std[1]:.3f}', ha='center', va='top',
                fontweight='bold', fontsize=10)
    
    # 添加说明
    axes[1].text(0, 0.3, 'Negative similarity\n= Strong protection', ha='center', va='top',
                fontsize=10, fontweight='bold', color=COLORS['success'],
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    # (c) Usability (STOI)
    bars3 = axes[2].bar(methods, stoi_mean, yerr=stoi_std, capsize=8,
                       color=colors, edgecolor='black', linewidth=1.5, alpha=0.85)
    axes[2].set_ylabel('STOI Score', fontweight='bold')
    axes[2].set_title('(c) Usability Preservation', fontweight='bold')
    axes[2].axhline(y=0.85, color='red', linestyle='--', linewidth=1.5, alpha=0.6, label='Threshold')
    axes[2].set_ylim([0.8, 1.0])
    axes[2].grid(axis='y', alpha=0.3)
    axes[2].legend()
    
    # 添加数值标签
    for bar, mean, std in zip(bars3, stoi_mean, stoi_std):
        height = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2., height + std + 0.005,
                    f'{mean:.3f}\n±{std:.3f}', ha='center', va='bottom',
                    fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    
    output_path = Path("paper/figures/fig7_optimization_comparison.pdf")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {output_path}")


def main():
    print("=" * 70)
    print("Regenerating Ablation Figures with Modern ML Colors")
    print("=" * 70)
    
    fig6_snr_ablation()
    fig7_optimization_comparison()
    
    print("\n" + "=" * 70)
    print("✓ All ablation figures regenerated successfully!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  paper/figures/fig6_snr_ablation.pdf")
    print("  paper/figures/fig7_optimization_comparison.pdf")


if __name__ == "__main__":
    main()

