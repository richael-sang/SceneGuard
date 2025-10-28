#!/usr/bin/env python3
"""
Baseline Comparison Script for SceneGuard Paper

Compares SceneGuard against random noise and Gaussian noise baselines.
Run on a subset of data (20 train + 10 test) for efficiency.
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

from src.sceneguard.eval.metrics import (
    compute_speaker_similarity,
    compute_wer,
    compute_pesq,
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

def add_random_noise(audio, snr_db_min=10, snr_db_max=20, sr=16000):
    """Add uniform random noise at random SNR in [snr_db_min, snr_db_max]"""
    # Sample random SNR
    snr_db = np.random.uniform(snr_db_min, snr_db_max)
    
    # Generate random noise
    noise = torch.rand_like(audio) * 2 - 1  # Uniform [-1, 1]
    
    # Compute RMS
    audio_rms = torch.sqrt(torch.mean(audio ** 2))
    noise_rms = torch.sqrt(torch.mean(noise ** 2))
    
    # Compute scaling factor
    alpha = (audio_rms / noise_rms) * (10 ** (-snr_db / 20))
    
    # Add noise
    noisy = audio + alpha * noise
    
    # Clip to valid range
    noisy = torch.clamp(noisy, -1.0, 1.0)
    
    return noisy

def add_gaussian_noise(audio, snr_db_min=10, snr_db_max=20, sr=16000):
    """Add zero-mean Gaussian noise at random SNR in [snr_db_min, snr_db_max]"""
    # Sample random SNR
    snr_db = np.random.uniform(snr_db_min, snr_db_max)
    
    # Generate Gaussian noise
    noise = torch.randn_like(audio)
    
    # Compute RMS
    audio_rms = torch.sqrt(torch.mean(audio ** 2))
    noise_rms = torch.sqrt(torch.mean(noise ** 2))
    
    # Compute scaling factor
    alpha = (audio_rms / noise_rms) * (10 ** (-snr_db / 20))
    
    # Add noise
    noisy = audio + alpha * noise
    
    # Clip to valid range
    noisy = torch.clamp(noisy, -1.0, 1.0)
    
    return noisy

def evaluate_training_quality(audio_paths, device='cuda'):
    """
    Evaluate training data quality using speaker embedding consistency.
    This is a proxy for TTS training quality.
    """
    from speechbrain.inference.speaker import EncoderClassifier
    
    print("Loading speaker verification model...")
    sv_model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="models/sv/ecapa-voxceleb",
        run_opts={"device": device}
    )
    
    embeddings = []
    for audio_path in tqdm(audio_paths, desc="Extracting embeddings"):
        audio, sr = load_audio(audio_path)
        audio = audio.to(device)
        with torch.no_grad():
            emb = sv_model.encode_batch(audio).squeeze().cpu()
        embeddings.append(emb)
    
    embeddings = torch.stack(embeddings)
    
    # Compute pairwise similarities
    sims = []
    for i in range(len(embeddings)):
        for j in range(i+1, len(embeddings)):
            sim = torch.nn.functional.cosine_similarity(
                embeddings[i].unsqueeze(0),
                embeddings[j].unsqueeze(0)
            ).item()
            sims.append(sim)
    
    return {
        'n_embeddings': len(embeddings),
        'mean_similarity': float(np.mean(sims)),
        'std_similarity': float(np.std(sims)),
        'embedding_dim': embeddings.shape[1]
    }

def evaluate_test_set(clean_paths, defended_paths, device='cuda'):
    """Evaluate speaker similarity on test set"""
    from speechbrain.inference.speaker import EncoderClassifier
    
    print("Loading speaker verification model...")
    sv_model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="models/sv/ecapa-voxceleb",
        run_opts={"device": device}
    )
    
    sims = []
    for clean_path, defended_path in tqdm(
        zip(clean_paths, defended_paths),
        total=len(clean_paths),
        desc="Computing similarities"
    ):
        sim = compute_speaker_similarity(clean_path, defended_path, sv_model)
        sims.append(sim)
    
    return np.array(sims)

def main():
    parser = argparse.ArgumentParser(description='Baseline comparison for SceneGuard')
    parser.add_argument('--n_train', type=int, default=20, help='Number of training samples')
    parser.add_argument('--n_test', type=int, default=10, help='Number of test samples')
    parser.add_argument('--seed', type=int, default=1337, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    # Setup paths
    base_dir = Path(__file__).parent.parent
    clean_dir = base_dir / 'data' / 'raw' / 'speech' / 'libritts' / '5339'
    output_dir = base_dir / 'results' / 'baselines'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    random_dir = output_dir / 'random_noise'
    gaussian_dir = output_dir / 'gaussian_noise'
    random_dir.mkdir(exist_ok=True)
    gaussian_dir.mkdir(exist_ok=True)
    
    # Get all audio files
    audio_files = sorted(list(clean_dir.glob('*.wav')))
    print(f"Found {len(audio_files)} audio files")
    
    # Split into train and test
    train_files = audio_files[:args.n_train]
    test_files = audio_files[args.n_train:args.n_train + args.n_test]
    
    print(f"Using {len(train_files)} training files and {len(test_files)} test files")
    
    # Generate random noise versions
    print("\n=== Generating Random Noise Baseline ===")
    random_train_files = []
    random_test_files = []
    
    for audio_file in tqdm(train_files + test_files, desc="Random noise"):
        audio, sr = load_audio(audio_file)
        noisy = add_random_noise(audio, sr=sr)
        
        out_path = random_dir / audio_file.name
        torchaudio.save(str(out_path), noisy, sr)
        
        if audio_file in train_files:
            random_train_files.append(out_path)
        else:
            random_test_files.append(out_path)
    
    # Generate Gaussian noise versions
    print("\n=== Generating Gaussian Noise Baseline ===")
    gaussian_train_files = []
    gaussian_test_files = []
    
    for audio_file in tqdm(train_files + test_files, desc="Gaussian noise"):
        audio, sr = load_audio(audio_file)
        noisy = add_gaussian_noise(audio, sr=sr)
        
        out_path = gaussian_dir / audio_file.name
        torchaudio.save(str(out_path), noisy, sr)
        
        if audio_file in train_files:
            gaussian_train_files.append(out_path)
        else:
            gaussian_test_files.append(out_path)
    
    # Evaluate training quality
    print("\n=== Evaluating Training Quality ===")
    print("Clean training data...")
    clean_train_quality = evaluate_training_quality([str(f) for f in train_files], args.device)
    
    print("Random noise training data...")
    random_train_quality = evaluate_training_quality([str(f) for f in random_train_files], args.device)
    
    print("Gaussian noise training data...")
    gaussian_train_quality = evaluate_training_quality([str(f) for f in gaussian_train_files], args.device)
    
    # Evaluate test set
    print("\n=== Evaluating Test Set ===")
    print("Random noise...")
    random_test_sims = evaluate_test_set(
        [str(f) for f in test_files],
        [str(f) for f in random_test_files],
        args.device
    )
    
    print("Gaussian noise...")
    gaussian_test_sims = evaluate_test_set(
        [str(f) for f in test_files],
        [str(f) for f in gaussian_test_files],
        args.device
    )
    
    # Compute statistics
    from scipy import stats
    
    def compute_stats(sims):
        # Compare to perfect similarity (1.0)
        t_stat, p_value = stats.ttest_1samp(sims, 1.0)
        cohens_d = (np.mean(sims) - 1.0) / np.std(sims, ddof=1)
        return {
            'mean': float(np.mean(sims)),
            'std': float(np.std(sims)),
            'p_value': float(p_value),
            'cohens_d': float(cohens_d)
        }
    
    random_stats = compute_stats(random_test_sims)
    gaussian_stats = compute_stats(gaussian_test_sims)
    
    # Compute WER, PESQ, STOI for quality metrics
    print("\n=== Computing Quality Metrics ===")
    import whisper
    whisper_model = whisper.load_model("base", device=args.device)
    
    def compute_quality_metrics(clean_paths, defended_paths):
        wers = []
        pesqs = []
        stois = []
        
        for clean_path, defended_path in tqdm(
            zip(clean_paths, defended_paths),
            total=len(clean_paths),
            desc="Quality metrics"
        ):
            wer = compute_wer(clean_path, defended_path, whisper_model, args.device)
            pesq = compute_pesq(clean_path, defended_path)
            stoi_val = compute_stoi(clean_path, defended_path)
            
            wers.append(wer)
            pesqs.append(pesq)
            stois.append(stoi_val)
        
        return {
            'wer_mean': float(np.mean(wers)),
            'wer_std': float(np.std(wers)),
            'pesq_mean': float(np.mean(pesqs)),
            'pesq_std': float(np.std(pesqs)),
            'stoi_mean': float(np.mean(stois)),
            'stoi_std': float(np.std(stois))
        }
    
    print("Random noise quality...")
    random_quality = compute_quality_metrics(
        [str(f) for f in test_files],
        [str(f) for f in random_test_files]
    )
    
    print("Gaussian noise quality...")
    gaussian_quality = compute_quality_metrics(
        [str(f) for f in test_files],
        [str(f) for f in gaussian_test_files]
    )
    
    # Compile results
    results = {
        'experiment_date': datetime.now().isoformat(),
        'n_train': args.n_train,
        'n_test': args.n_test,
        'seed': args.seed,
        'clean_train_quality': clean_train_quality,
        'random_noise': {
            'train_quality': random_train_quality,
            'test_similarity': random_stats,
            'quality_metrics': random_quality
        },
        'gaussian_noise': {
            'train_quality': gaussian_train_quality,
            'test_similarity': gaussian_stats,
            'quality_metrics': gaussian_quality
        }
    }
    
    # Save results
    summary_path = output_dir / 'baseline_comparison_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {summary_path}")
    
    # Print summary
    print("\n" + "="*70)
    print("BASELINE COMPARISON SUMMARY")
    print("="*70)
    print(f"\nTraining Data Quality (mean pairwise similarity):")
    print(f"  Clean:    {clean_train_quality['mean_similarity']:.4f}")
    print(f"  Random:   {random_train_quality['mean_similarity']:.4f}")
    print(f"  Gaussian: {gaussian_train_quality['mean_similarity']:.4f}")
    
    print(f"\nTest Set Similarity (clean vs defended):")
    print(f"  Random:   {random_stats['mean']:.4f} (p={random_stats['p_value']:.2e}, d={random_stats['cohens_d']:.2f})")
    print(f"  Gaussian: {gaussian_stats['mean']:.4f} (p={gaussian_stats['p_value']:.2e}, d={gaussian_stats['cohens_d']:.2f})")
    
    print(f"\nQuality Metrics:")
    print(f"  Random   - WER: {random_quality['wer_mean']*100:.2f}%, PESQ: {random_quality['pesq_mean']:.2f}, STOI: {random_quality['stoi_mean']:.3f}")
    print(f"  Gaussian - WER: {gaussian_quality['wer_mean']*100:.2f}%, PESQ: {gaussian_quality['pesq_mean']:.2f}, STOI: {gaussian_quality['stoi_mean']:.3f}")
    print("="*70)

if __name__ == '__main__':
    main()

