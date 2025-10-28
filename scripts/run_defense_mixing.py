#!/usr/bin/env python3
"""
SceneGuard T50: Defense Mixing
Generate protected speech by mixing scene-consistent noise

Supports two modes:
- direct: Direct mixing with random SNR and stochastic mask (baseline)
- optimized: Gradient-based optimization of mask m(t) and strength γ
"""
import sys
import json
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import random
import numpy as np
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sceneguard.mixing import SceneGuardMixer


def run_defense_mixing(mode='direct', seed=1337, max_epochs=50, lambda_sim=1.0, lambda_reg=0.01, lr=0.01):
    """
    Main defense mixing function
    
    Args:
        mode: 'direct' or 'optimized'
        seed: Random seed
        max_epochs: Maximum optimization epochs (optimized mode only)
        lambda_sim: Weight for speaker similarity loss (optimized mode only)
        lambda_reg: Weight for regularization (optimized mode only)
        lr: Learning rate (optimized mode only)
    """
    print("=" * 70)
    print(f"SceneGuard T50: Defense Mixing (Mode: {mode.upper()})")
    print("=" * 70)
    
    # Set seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Load scene labels
    print("\n[1/6] Loading scene labels...")
    labels_file = Path("data/interim/scene_labels.csv")
    labels_df = pd.read_csv(labels_file)
    print(f"  Loaded: {len(labels_df)} audio files")
    print(f"  Scenes: {labels_df['scene_label'].nunique()} unique")
    
    # Load speaker encoder if optimized mode
    speaker_encoder = None
    if mode == 'optimized':
        print("\n[2/6] Loading ECAPA-TDNN speaker encoder...")
        try:
            from speechbrain.inference import EncoderClassifier
        except ImportError:
            from speechbrain.pretrained import EncoderClassifier
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"  Device: {device}")
        
        speaker_encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="models/sv/ecapa/",
            run_opts={"device": device}
        )
        print("  ✓ Loaded ECAPA-TDNN")
    else:
        print("\n[2/6] Skipping model loading (direct mode)")
    
    # Initialize mixer
    print(f"\n[3/6] Initializing SceneGuard Mixer...")
    mixer = SceneGuardMixer(
        noise_taxonomy_path="artifacts/noise_taxonomy.json",
        snr_range=(10.0, 20.0),
        masking_strategy="stochastic",
        target_sr=16000,
        seed=seed
    )
    
    # Setup output directories
    if mode == 'direct':
        output_dir = Path("data/processed/defended/5339")
        params_file = Path("artifacts/mix_params.jsonl")
    else:
        output_dir = Path("data/processed/defended_optimized/5339")
        params_file = Path("artifacts/mix_params_optimized.jsonl")
        
        # Create optimization logs directory
        opt_logs_dir = Path("artifacts/optimization_logs")
        opt_logs_dir.mkdir(parents=True, exist_ok=True)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process all files
    print(f"\n[4/6] Mixing {len(labels_df)} audio files...")
    print(f"  Output: {output_dir}")
    if mode == 'optimized':
        print(f"  Optimization: epochs={max_epochs}, λ_SIM={lambda_sim}, λ_REG={lambda_reg}, lr={lr}")
    
    mix_params_list = []
    failed = []
    
    for idx, row in tqdm(labels_df.iterrows(), total=len(labels_df), desc="  Processing"):
        try:
            speech_path = row['audio_path']
            scene = row['scene_label']
            
            # Generate output filename
            filename = Path(speech_path).name
            output_path = output_dir / filename
            
            # Mix based on mode
            if mode == 'direct':
                mixed, params = mixer.mix(
                    speech_path=speech_path,
                    scene=scene,
                    output_path=str(output_path)
                )
            else:  # optimized
                log_path = opt_logs_dir / f"{Path(speech_path).stem}_opt.json"
                
                mixed, params = mixer.mix_optimized(
                    speech_path=speech_path,
                    scene=scene,
                    speaker_encoder=speaker_encoder,
                    output_path=str(output_path),
                    max_epochs=max_epochs,
                    lambda_sim=lambda_sim,
                    lambda_reg=lambda_reg,
                    lr=lr,
                    log_path=str(log_path)
                )
            
            # Log parameters
            params['index'] = idx
            params['original_path'] = speech_path
            params['mode'] = mode
            mix_params_list.append(params)
            
        except Exception as e:
            print(f"\n  ✗ Failed on {speech_path}: {e}")
            import traceback
            traceback.print_exc()
            failed.append(speech_path)
    
    # Save mixing parameters
    print(f"\n[5/6] Saving mixing parameters...")
    with open(params_file, 'w') as f:
        for params in mix_params_list:
            f.write(json.dumps(params) + '\n')
    print(f"  Saved to: {params_file}")
    print(f"  Total: {len(mix_params_list)} entries")
    
    # Statistics
    print(f"\n[6/6] Mixing statistics...")
    params_df = pd.DataFrame(mix_params_list)
    
    print(f"  Processed: {len(mix_params_list)}/{len(labels_df)} files")
    print(f"  Failed: {len(failed)}")
    
    if mode == 'direct':
        print(f"\n  SNR statistics:")
        print(f"    Mean: {params_df['snr_db'].mean():.2f} dB")
        print(f"    Std:  {params_df['snr_db'].std():.2f} dB")
        print(f"    Range: [{params_df['snr_db'].min():.2f}, {params_df['snr_db'].max():.2f}] dB")
    else:
        print(f"\n  Final SNR statistics:")
        print(f"    Mean: {params_df['final_snr_db'].mean():.2f} dB")
        print(f"    Std:  {params_df['final_snr_db'].std():.2f} dB")
        print(f"    Range: [{params_df['final_snr_db'].min():.2f}, {params_df['final_snr_db'].max():.2f}] dB")
        
        print(f"\n  Final speaker similarity:")
        print(f"    Mean: {params_df['final_similarity'].mean():.4f}")
        print(f"    Std:  {params_df['final_similarity'].std():.4f}")
        print(f"    Range: [{params_df['final_similarity'].min():.4f}, {params_df['final_similarity'].max():.4f}]")
        
        print(f"\n  Mask statistics:")
        print(f"    Mean: {params_df['mask_mean'].mean():.4f} ± {params_df['mask_mean'].std():.4f}")
        print(f"    Std:  {params_df['mask_std'].mean():.4f} ± {params_df['mask_std'].std():.4f}")
    
    print(f"\n  RMS statistics:")
    print(f"    Speech: {params_df['speech_rms'].mean():.4f} ± {params_df['speech_rms'].std():.4f}")
    print(f"    Noise:  {params_df['noise_rms'].mean():.4f} ± {params_df['noise_rms'].std():.4f}")
    print(f"    Mixed:  {params_df['mixed_rms'].mean():.4f} ± {params_df['mixed_rms'].std():.4f}")
    
    # Scene distribution
    print(f"\n  Scene distribution:")
    scene_counts = params_df['scene'].value_counts()
    for scene, count in scene_counts.items():
        print(f"    {scene:20s}: {count:3d} files")
    
    # Validation
    print("\n" + "=" * 70)
    print("Validation")
    print("=" * 70)
    
    # Check output files
    output_files = list(output_dir.glob("*.wav"))
    
    checks = [
        ("All files processed", len(mix_params_list) == len(labels_df), 
         f"{len(mix_params_list)}/{len(labels_df)} files"),
        ("No failures", len(failed) == 0, 
         f"{len(failed)} failures"),
        ("Output files exist", len(output_files) == len(labels_df), 
         f"{len(output_files)} files in output dir"),
        ("Params logged", len(mix_params_list) == len(output_files), 
         f"{len(mix_params_list)} params logged"),
    ]
    
    all_pass = True
    for check_name, passed, message in checks:
        status = "✓" if passed else "✗"
        print(f"  {status} {check_name}: {message}")
        if not passed:
            all_pass = False
    
    if all_pass:
        print(f"\n✓ Defense mixing ({mode}) complete and validated!")
        print(f"\nDefended audio saved to: {output_dir}")
        print(f"Mix parameters saved to: {params_file}")
        if mode == 'optimized':
            print(f"Optimization logs saved to: {opt_logs_dir}")
        return 0
    else:
        print("\n✗ Validation failed")
        if failed:
            print(f"\nFailed files:")
            for f in failed:
                print(f"  - {f}")
        return 1


def main():
    parser = argparse.ArgumentParser(
        description='SceneGuard Defense Mixing',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['direct', 'optimized'],
        default='direct',
        help='Mixing mode: direct (baseline) or optimized (gradient-based)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=1337,
        help='Random seed for reproducibility'
    )
    
    # Optimization parameters (only for optimized mode)
    parser.add_argument(
        '--max-epochs',
        type=int,
        default=50,
        help='Maximum optimization epochs (optimized mode only)'
    )
    
    parser.add_argument(
        '--lambda-sim',
        type=float,
        default=1.0,
        help='Weight for speaker similarity loss (optimized mode only)'
    )
    
    parser.add_argument(
        '--lambda-reg',
        type=float,
        default=0.01,
        help='Weight for regularization (optimized mode only)'
    )
    
    parser.add_argument(
        '--lr',
        type=float,
        default=0.01,
        help='Learning rate (optimized mode only)'
    )
    
    args = parser.parse_args()
    
    return run_defense_mixing(
        mode=args.mode,
        seed=args.seed,
        max_epochs=args.max_epochs,
        lambda_sim=args.lambda_sim,
        lambda_reg=args.lambda_reg,
        lr=args.lr
    )


if __name__ == "__main__":
    sys.exit(main())
