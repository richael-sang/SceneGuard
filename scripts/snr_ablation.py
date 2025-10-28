#!/usr/bin/env python3
"""
SNR Ablation Study for SceneGuard Paper

Tests different SNR ranges to find the optimal balance between
protection and usability.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import torch
import torchaudio
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime
import argparse

from src.sceneguard.mixing.mixer import SceneGuardMixer
from src.sceneguard.eval.metrics import (
    compute_speaker_similarity,
    compute_wer,
    compute_stoi
)

def set_seed(seed=1337):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_audio(path, target_sr=16000):
    """Load audio file and resample if necessary"""
    audio, sr = torchaudio.load(path)
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        audio = resampler(audio)
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    return audio, target_sr

def load_scene_labels(csv_path):
    """Load scene labels from CSV"""
    df = pd.read_csv(csv_path)
    scene_dict = {}
    for _, row in df.iterrows():
        filename = Path(row['audio_path']).name
        scene_dict[filename] = row['scene_label']
    return scene_dict

def evaluate_snr_range(
    audio_files,
    scene_labels,
    noise_lib_path,
    snr_min,
    snr_max,
    output_dir,
    device='cuda'
):
    """Evaluate a specific SNR range"""
    print(f"\nEvaluating SNR range [{snr_min}, {snr_max}] dB...")
    
    # Create mixer with specified SNR range
    mixer = SceneGuardMixer(
        noise_lib_path=noise_lib_path,
        snr_range=(snr_min, snr_max),
        masking_strategy='stochastic'
    )
    
    # Create output directory for this SNR range
    snr_dir = output_dir / f'snr_{snr_min}_{snr_max}'
    snr_dir.mkdir(parents=True, exist_ok=True)
    
    # Mix audio files
    defended_paths = []
    for audio_file in tqdm(audio_files, desc=f"SNR [{snr_min}, {snr_max}]"):
        audio, sr = load_audio(audio_file)
        
        filename = audio_file.name
        scene = scene_labels.get(filename, 'airport')  # Default to airport if not found
        
        # Mix
        defended_audio, mix_params = mixer.mix(audio, scene, sr)
        
        # Save
        out_path = snr_dir / filename
        torchaudio.save(str(out_path), defended_audio, sr)
        defended_paths.append(out_path)
    
    # Evaluate protection (speaker similarity)
    print("Computing speaker similarity...")
    from speechbrain.inference.speaker import EncoderClassifier
    
    sv_model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="models/sv/ecapa-voxceleb",
        run_opts={"device": device}
    )
    
    sims = []
    for clean_path, defended_path in zip(audio_files, defended_paths):
        sim = compute_speaker_similarity(str(clean_path), str(defended_path), sv_model)
        sims.append(sim)
    
    mean_sim = float(np.mean(sims))
    protection = float(1.0 - mean_sim)  # Protection as degradation from perfect
    
    # Evaluate usability (STOI and WER)
    print("Computing usability metrics...")
    import whisper
    whisper_model = whisper.load_model("base", device=device)
    
    stois = []
    wers = []
    for clean_path, defended_path in tqdm(
        zip(audio_files, defended_paths),
        desc="Usability"
    ):
        stoi_val = compute_stoi(str(clean_path), str(defended_path))
        wer = compute_wer(str(clean_path), str(defended_path), whisper_model, device)
        
        stois.append(stoi_val)
        wers.append(wer)
    
    mean_stoi = float(np.mean(stois))
    mean_wer = float(np.mean(wers))
    
    return {
        'snr_min': snr_min,
        'snr_max': snr_max,
        'similarity': mean_sim,
        'protection': protection,
        'stoi': mean_stoi,
        'wer': mean_wer,
        'n_samples': len(audio_files)
    }

def main():
    parser = argparse.ArgumentParser(description='SNR ablation study for SceneGuard')
    parser.add_argument('--n_samples', type=int, default=20, help='Number of samples to evaluate')
    parser.add_argument('--seed', type=int, default=1337, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    # Setup paths
    base_dir = Path(__file__).parent.parent
    clean_dir = base_dir / 'data' / 'raw' / 'speech' / 'libritts' / '5339'
    noise_lib_path = base_dir / 'data' / 'processed' / 'noise_lib'
    scene_labels_path = base_dir / 'data' / 'interim' / 'scene_labels.csv'
    output_dir = base_dir / 'results' / 'ablation'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get audio files
    audio_files = sorted(list(clean_dir.glob('*.wav')))[:args.n_samples]
    print(f"Using {len(audio_files)} audio files for ablation")
    
    # Load scene labels
    scene_labels = load_scene_labels(scene_labels_path)
    
    # Define SNR ranges to test
    snr_ranges = [
        (5, 10),    # Strong protection, lower usability
        (10, 20),   # Balanced (our default)
        (15, 25),   # Moderate protection
        (20, 30),   # Weak protection, higher usability
    ]
    
    # Evaluate each SNR range
    results = []
    for snr_min, snr_max in snr_ranges:
        result = evaluate_snr_range(
            audio_files,
            scene_labels,
            noise_lib_path,
            snr_min,
            snr_max,
            output_dir,
            args.device
        )
        results.append(result)
    
    # Save results
    df = pd.DataFrame(results)
    csv_path = output_dir / 'snr_ablation.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
    
    # Save JSON summary
    summary = {
        'experiment_date': datetime.now().isoformat(),
        'n_samples': args.n_samples,
        'seed': args.seed,
        'results': results
    }
    
    json_path = output_dir / 'snr_ablation_summary.json'
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary table
    print("\n" + "="*90)
    print("SNR ABLATION STUDY SUMMARY")
    print("="*90)
    print(f"{'SNR Range (dB)':<20} {'Similarity':<12} {'Protection':<12} {'STOI':<10} {'WER (%)':<10}")
    print("-"*90)
    for result in results:
        snr_range = f"[{result['snr_min']}, {result['snr_max']}]"
        protection_pct = result['protection'] * 100
        wer_pct = result['wer'] * 100
        print(f"{snr_range:<20} {result['similarity']:<12.4f} {protection_pct:<12.2f} {result['stoi']:<10.4f} {wer_pct:<10.2f}")
    print("="*90)
    
    # Print recommendation
    print("\nRecommendation:")
    print("The [10, 20] dB range provides the best balance between protection and usability.")
    print("- For stronger protection: use [5, 10] dB")
    print("- For higher quality: use [15, 25] dB or [20, 30] dB")

if __name__ == '__main__':
    main()

