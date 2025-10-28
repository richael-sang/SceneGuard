#!/usr/bin/env python3
"""
分析Direct vs Optimized混音的对比结果
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path

def load_mix_params(jsonl_path):
    """加载混音参数"""
    data = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame(data)

def analyze_comparison():
    """对比direct和optimized模式"""
    
    print("=" * 70)
    print("Direct vs Optimized Mixing Comparison")
    print("=" * 70)
    
    # Load data
    direct_file = Path("artifacts/mix_params.jsonl")
    optimized_file = Path("artifacts/mix_params_optimized.jsonl")
    
    if not direct_file.exists():
        print(f"\n⚠ Direct mixing results not found: {direct_file}")
        print("  Run: python scripts/run_defense_mixing.py --mode direct")
        return
    
    if not optimized_file.exists():
        print(f"\n✗ Optimized mixing results not found: {optimized_file}")
        return
    
    df_direct = load_mix_params(direct_file)
    df_optimized = load_mix_params(optimized_file)
    
    print(f"\n[1/3] Data loaded")
    print(f"  Direct samples:    {len(df_direct)}")
    print(f"  Optimized samples: {len(df_optimized)}")
    
    # Compare SNR
    print(f"\n[2/3] SNR Comparison")
    print(f"  Direct:")
    print(f"    Mean: {df_direct['snr_db'].mean():.2f} ± {df_direct['snr_db'].std():.2f} dB")
    print(f"    Range: [{df_direct['snr_db'].min():.2f}, {df_direct['snr_db'].max():.2f}] dB")
    
    print(f"  Optimized:")
    print(f"    Mean: {df_optimized['final_snr_db'].mean():.2f} ± {df_optimized['final_snr_db'].std():.2f} dB")
    print(f"    Range: [{df_optimized['final_snr_db'].min():.2f}, {df_optimized['final_snr_db'].max():.2f}] dB")
    
    # Compare Mask
    print(f"\n[3/3] Mask Comparison")
    print(f"  Direct: (stochastic, no optimization)")
    
    print(f"  Optimized:")
    print(f"    Mean: {df_optimized['mask_mean'].mean():.4f} ± {df_optimized['mask_mean'].std():.4f}")
    print(f"    Std:  {df_optimized['mask_std'].mean():.4f} ± {df_optimized['mask_std'].std():.4f}")
    
    # Similarity (only for optimized)
    if 'final_similarity' in df_optimized.columns:
        print(f"\n  Speaker Similarity (Optimized only):")
        print(f"    Mean: {df_optimized['final_similarity'].mean():.4f} ± {df_optimized['final_similarity'].std():.4f}")
        print(f"    Range: [{df_optimized['final_similarity'].min():.4f}, {df_optimized['final_similarity'].max():.4f}]")
    
    # Generate summary for paper
    print("\n" + "=" * 70)
    print("Paper Update Summary")
    print("=" * 70)
    
    print("\n📊 Table 6.2 Data (Optimization Ablation):")
    print("Direct Mixing:")
    print(f"  SNR: {df_direct['snr_db'].mean():.1f} ± {df_direct['snr_db'].std():.1f} dB")
    print(f"  (Note: Need to run evaluation to get SIM/STOI/WER)")
    
    print("\nOptimized (Ours):")
    print(f"  SNR: {df_optimized['final_snr_db'].mean():.2f} ± {df_optimized['final_snr_db'].std():.2f} dB")
    print(f"  Speaker Similarity: {df_optimized['final_similarity'].mean():.3f} ± {df_optimized['final_similarity'].std():.3f}")
    print(f"  Mask Mean: {df_optimized['mask_mean'].mean():.4f}")
    print(f"  Mask Std:  {df_optimized['mask_std'].mean():.4f}")
    
    print("\n📝 Key Findings:")
    print(f"  • SNR constraint satisfied: {(df_optimized['final_snr_db'] >= 10).all() and (df_optimized['final_snr_db'] <= 20).all()}")
    print(f"  • Tight SNR clustering: σ = {df_optimized['final_snr_db'].std():.4f} dB")
    print(f"  • Consistent mask patterns: σ_mean = {df_optimized['mask_mean'].std():.4f}")
    print(f"  • Strong protection: mean similarity = {df_optimized['final_similarity'].mean():.3f} (negative = strong protection)")
    
    # Save comparison data
    output_file = Path("artifacts/optimization_comparison_summary.json")
    summary = {
        "direct": {
            "n_samples": len(df_direct),
            "snr_mean": float(df_direct['snr_db'].mean()),
            "snr_std": float(df_direct['snr_db'].std()),
            "snr_min": float(df_direct['snr_db'].min()),
            "snr_max": float(df_direct['snr_db'].max()),
        },
        "optimized": {
            "n_samples": len(df_optimized),
            "snr_mean": float(df_optimized['final_snr_db'].mean()),
            "snr_std": float(df_optimized['final_snr_db'].std()),
            "snr_min": float(df_optimized['final_snr_db'].min()),
            "snr_max": float(df_optimized['final_snr_db'].max()),
            "similarity_mean": float(df_optimized['final_similarity'].mean()),
            "similarity_std": float(df_optimized['final_similarity'].std()),
            "similarity_min": float(df_optimized['final_similarity'].min()),
            "similarity_max": float(df_optimized['final_similarity'].max()),
            "mask_mean_avg": float(df_optimized['mask_mean'].mean()),
            "mask_mean_std": float(df_optimized['mask_mean'].std()),
            "mask_std_avg": float(df_optimized['mask_std'].mean()),
            "mask_std_std": float(df_optimized['mask_std'].std()),
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Summary saved to: {output_file}")

if __name__ == "__main__":
    analyze_comparison()

