#!/usr/bin/env python3
"""
SceneGuard T30: Build Scene-Organized Noise Library
Processes TAU dataset to create clean noise samples organized by scene
"""
import os
import json
import pandas as pd
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Configuration
TARGET_SR = 16000  # Target sampling rate for all audio
MIN_CLIPS_PER_SCENE = 50  # Minimum clips required per scene
MAX_CLIPS_PER_SCENE = 5000  # Maximum to process (for speed)
TARGET_RMS = 0.01  # Target RMS for normalization

def load_and_resample(audio_path, target_sr=16000):
    """Load audio and resample to target SR"""
    try:
        # Load audio
        audio, sr = librosa.load(audio_path, sr=None, mono=True)
        
        # Resample if needed
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        
        return audio, target_sr
    except Exception as e:
        print(f"  Error loading {audio_path}: {e}")
        return None, None

def rms_normalize(audio, target_rms=0.01):
    """Normalize audio to target RMS level"""
    current_rms = np.sqrt(np.mean(audio**2))
    if current_rms > 0:
        audio = audio * (target_rms / current_rms)
    return audio

def has_speech_content(audio, sr, threshold_energy=0.001):
    """
    Simple speech detection: check if audio has sufficient energy
    More sophisticated: would use Whisper WER, but too slow for 230k files
    """
    # Check energy distribution
    rms = np.sqrt(np.mean(audio**2))
    
    # Very low energy = likely pure noise
    if rms < threshold_energy:
        return False
    
    # For now, accept all TAU files (they're environmental sounds)
    # Future: run Whisper on subset to filter speech
    return False  # Assume TAU has no speech (it's environmental sounds)

def process_tau_dataset():
    """Main processing function"""
    print("=" * 70)
    print("SceneGuard T30: Building Scene-Organized Noise Library")
    print("=" * 70)
    
    # Paths
    meta_file = Path("data/TAU-urban-acoustic-scenes-2022-mobile-development/development/meta.csv")
    audio_base = Path("data/TAU-urban-acoustic-scenes-2022-mobile-development/development")
    output_base = Path("data/processed/noise_lib")
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Load metadata
    print("\n[1/5] Loading TAU metadata...")
    meta = pd.read_csv(meta_file, sep='\t')
    print(f"  Total files: {len(meta)}")
    print(f"  Scenes: {meta['scene_label'].unique().tolist()}")
    
    # Scene distribution
    scene_counts = meta['scene_label'].value_counts()
    print(f"\n[2/5] Scene distribution:")
    for scene, count in scene_counts.items():
        print(f"  {scene:20s}: {count:,} files")
    
    # Process by scene
    print(f"\n[3/5] Processing audio files...")
    print(f"  Target SR: {TARGET_SR} Hz")
    print(f"  Max clips per scene: {MAX_CLIPS_PER_SCENE}")
    
    taxonomy = defaultdict(list)
    statistics = []
    
    for scene in tqdm(meta['scene_label'].unique(), desc="Scenes"):
        scene_dir = output_base / scene
        scene_dir.mkdir(exist_ok=True)
        
        # Get files for this scene
        scene_files = meta[meta['scene_label'] == scene]['filename'].tolist()
        
        # Limit number of files to process
        scene_files = scene_files[:MAX_CLIPS_PER_SCENE]
        
        processed_count = 0
        total_duration = 0
        rms_values = []
        
        for filename in tqdm(scene_files, desc=f"  {scene}", leave=False, disable=True):
            audio_path = audio_base / filename
            
            # Load and resample
            audio, sr = load_and_resample(audio_path, TARGET_SR)
            if audio is None:
                continue
            
            # Check for speech (simple version - assume TAU is clean)
            # if has_speech_content(audio, sr):
            #     continue
            
            # RMS normalize
            audio_norm = rms_normalize(audio, TARGET_RMS)
            
            # Save processed audio
            output_filename = Path(filename).stem + "_processed.wav"
            output_path = scene_dir / output_filename
            sf.write(output_path, audio_norm, TARGET_SR)
            
            # Track statistics
            taxonomy[scene].append(str(output_path))
            total_duration += len(audio) / sr
            rms_values.append(np.sqrt(np.mean(audio_norm**2)))
            processed_count += 1
        
        # Scene statistics
        statistics.append({
            'scene': scene,
            'num_clips': processed_count,
            'total_duration_sec': total_duration,
            'mean_duration_sec': total_duration / processed_count if processed_count > 0 else 0,
            'mean_rms': np.mean(rms_values) if rms_values else 0,
            'std_rms': np.std(rms_values) if rms_values else 0
        })
        
        print(f"  {scene:20s}: {processed_count}/{len(scene_files)} processed")
    
    # Save taxonomy
    print(f"\n[4/5] Saving noise taxonomy...")
    taxonomy_dict = {k: v for k, v in taxonomy.items()}
    with open('artifacts/noise_taxonomy.json', 'w') as f:
        json.dump(taxonomy_dict, f, indent=2)
    print(f"  Saved to: artifacts/noise_taxonomy.json")
    
    # Save statistics
    print(f"\n[5/5] Saving statistics...")
    stats_df = pd.DataFrame(statistics)
    stats_df.to_csv('artifacts/noise_statistics.csv', index=False)
    print(f"  Saved to: artifacts/noise_statistics.csv")
    
    # Summary
    print("\n" + "=" * 70)
    print("Noise Library Summary")
    print("=" * 70)
    for _, row in stats_df.iterrows():
        print(f"{row['scene']:20s}: {row['num_clips']:5.0f} clips, "
              f"{row['total_duration_sec']/60:6.1f} min, "
              f"RMS={row['mean_rms']:.4f}±{row['std_rms']:.4f}")
    
    # Validation
    print("\n" + "=" * 70)
    print("Validation")
    print("=" * 70)
    all_pass = True
    for _, row in stats_df.iterrows():
        if row['num_clips'] < MIN_CLIPS_PER_SCENE:
            print(f"  ✗ {row['scene']}: Only {row['num_clips']} clips (need {MIN_CLIPS_PER_SCENE})")
            all_pass = False
        else:
            print(f"  ✓ {row['scene']}: {row['num_clips']} clips")
    
    if all_pass:
        print("\n✓ All scenes have sufficient clips!")
        return 0
    else:
        print("\n✗ Some scenes need more clips")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(process_tau_dataset())

