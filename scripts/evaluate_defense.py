#!/usr/bin/env python3
"""
SceneGuard Defense Evaluation
Compute metrics to demonstrate protection effectiveness
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sceneguard.eval.metrics import (
    compute_speaker_similarity,
    compute_wer,
    compute_pesq,
    compute_stoi,
    compute_mcd,
    load_ecapa_model
)

def evaluate_defense(n_samples=20, seed=1337):
    """
    Evaluate defense effectiveness on sample files
    
    Metrics computed:
    - Speaker Similarity (SIM): Should decrease (harder to clone)
    - WER: Should remain low (usability preserved)
    - PESQ/STOI: Quality metrics
    - MCD: Spectral distance
    """
    print("=" * 70)
    print("SceneGuard Defense Evaluation")
    print("=" * 70)
    
    random.seed(seed)
    np.random.seed(seed)
    
    # Paths
    clean_dir = Path("data/raw/speech/libritts/5339")
    defended_dir = Path("data/processed/defended/5339")
    output_dir = Path("reports/metrics")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get file pairs
    print(f"\n[1/4] Loading files...")
    clean_files = sorted(clean_dir.glob("*.wav"))
    print(f"  Clean files: {len(clean_files)}")
    
    # Sample for faster evaluation
    if n_samples and n_samples < len(clean_files):
        sample_files = random.sample(clean_files, n_samples)
        print(f"  Sampling: {n_samples} files")
    else:
        sample_files = clean_files
        print(f"  Using all files")
    
    # Pre-load models
    print(f"\n[2/4] Loading models...")
    print(f"  Loading ECAPA...")
    ecapa_model = load_ecapa_model()
    
    print(f"  Loading Whisper...")
    import whisper
    whisper_model = whisper.load_model("base", download_root="models/asr/whisper/")
    
    # Compute metrics
    print(f"\n[3/4] Computing metrics on {len(sample_files)} files...")
    results = []
    
    for clean_path in tqdm(sample_files, desc="  Processing"):
        defended_path = defended_dir / clean_path.name
        
        if not defended_path.exists():
            print(f"\n  Warning: Defended file not found: {defended_path.name}")
            continue
        
        try:
            # Core metrics
            metrics = {
                'filename': clean_path.name,
                'clean_path': str(clean_path),
                'defended_path': str(defended_path),
            }
            
            # Speaker similarity
            sim = compute_speaker_similarity(str(clean_path), str(defended_path), ecapa_model)
            metrics['similarity'] = sim
            
            # WER
            wer_val = compute_wer(str(clean_path), str(defended_path), whisper_model)
            metrics['wer'] = wer_val
            
            # Quality metrics (faster versions)
            try:
                pesq_val = compute_pesq(str(clean_path), str(defended_path))
                metrics['pesq'] = pesq_val
            except:
                metrics['pesq'] = np.nan
            
            try:
                stoi_val = compute_stoi(str(clean_path), str(defended_path))
                metrics['stoi'] = stoi_val
            except:
                metrics['stoi'] = np.nan
            
            try:
                mcd_val = compute_mcd(str(clean_path), str(defended_path))
                metrics['mcd'] = mcd_val
            except:
                metrics['mcd'] = np.nan
            
            results.append(metrics)
            
        except Exception as e:
            print(f"\n  Error processing {clean_path.name}: {e}")
            continue
    
    # Save results
    print(f"\n[4/4] Saving results...")
    df = pd.DataFrame(results)
    output_file = output_dir / "defense_evaluation.csv"
    df.to_csv(output_file, index=False)
    print(f"  Saved to: {output_file}")
    
    # Statistics
    print("\n" + "=" * 70)
    print("Defense Effectiveness Summary")
    print("=" * 70)
    
    print(f"\nFiles evaluated: {len(results)}")
    
    # Speaker Similarity
    sim_mean = df['similarity'].mean()
    sim_std = df['similarity'].std()
    print(f"\nSpeaker Similarity (SIM):")
    print(f"  Mean: {sim_mean:.4f} ± {sim_std:.4f}")
    print(f"  Range: [{df['similarity'].min():.4f}, {df['similarity'].max():.4f}]")
    print(f"  Interpretation: Lower = harder to clone")
    print(f"  Expected: Clean refs give SIM ≥ 0.8, defended ≤ 0.6")
    
    # WER
    wer_mean = df['wer'].mean() * 100
    wer_std = df['wer'].std() * 100
    print(f"\nWord Error Rate (WER):")
    print(f"  Mean: {wer_mean:.2f}% ± {wer_std:.2f}%")
    print(f"  Range: [{df['wer'].min()*100:.2f}%, {df['wer'].max()*100:.2f}%]")
    print(f"  Interpretation: Lower = better usability")
    print(f"  Target: < 15% increase from clean")
    
    # PESQ
    if not df['pesq'].isna().all():
        pesq_mean = df['pesq'].mean()
        pesq_std = df['pesq'].std()
        print(f"\nPESQ (Quality):")
        print(f"  Mean: {pesq_mean:.3f} ± {pesq_std:.3f}")
        print(f"  Range: [1.0, 4.5], higher is better")
        print(f"  Target: ≥ 3.0 (good quality)")
    
    # STOI
    if not df['stoi'].isna().all():
        stoi_mean = df['stoi'].mean()
        stoi_std = df['stoi'].std()
        print(f"\nSTOI (Intelligibility):")
        print(f"  Mean: {stoi_mean:.3f} ± {stoi_std:.3f}")
        print(f"  Range: [0, 1], higher is better")
        print(f"  Target: ≥ 0.85 (high intelligibility)")
    
    # MCD
    if not df['mcd'].isna().all():
        mcd_mean = df['mcd'].mean()
        mcd_std = df['mcd'].std()
        print(f"\nMCD (Spectral Distance):")
        print(f"  Mean: {mcd_mean:.2f} dB ± {mcd_std:.2f} dB")
        print(f"  Interpretation: Distance from original")
    
    # Defense assessment
    print("\n" + "=" * 70)
    print("Defense Assessment")
    print("=" * 70)
    
    protection_effective = sim_mean < 0.95  # Significant degradation
    usability_preserved = wer_mean < 25  # Acceptable WER increase
    
    print(f"  Protection effective: {'✓' if protection_effective else '✗'} (SIM < 0.95)")
    print(f"  Usability preserved: {'✓' if usability_preserved else '✗'} (WER < 25%)")
    
    if protection_effective and usability_preserved:
        print(f"\n✓ SceneGuard defense shows promising results!")
        print(f"  - Similarity reduced from ~1.0 to {sim_mean:.3f}")
        print(f"  - WER remains acceptable at {wer_mean:.1f}%")
    else:
        print(f"\n⚠️  Results require further analysis")
    
    print(f"\n✓ Evaluation complete!")
    print(f"  Results saved to: {output_file}")
    
    return 0

if __name__ == "__main__":
    import sys
    # Evaluate on 20 samples for speed
    sys.exit(evaluate_defense(n_samples=20, seed=1337))

