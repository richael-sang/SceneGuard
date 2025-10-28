#!/usr/bin/env python3
"""
SceneGuard T40: Scene Labeling
Assigns acoustic scene labels to speech files for noise matching
"""
import hashlib
import pandas as pd
from pathlib import Path
import random
from tqdm import tqdm

# TAU scene classes
TAU_SCENES = [
    'airport', 'bus', 'metro', 'metro_station', 'park',
    'public_square', 'shopping_mall', 'street_pedestrian', 
    'street_traffic', 'tram'
]

def assign_scene_heuristic(audio_path, scenes, seed=1337):
    """
    Deterministic scene assignment based on file hash
    
    This ensures:
    - Consistent assignment across runs
    - Even distribution across scenes
    - No ASC model download needed for prototyping
    
    Args:
        audio_path: Path to audio file
        scenes: List of scene labels
        seed: Random seed for reproducibility
    
    Returns:
        scene_label: Assigned scene string
    """
    # Use file path hash for deterministic assignment
    file_hash = hashlib.md5(str(audio_path).encode()).hexdigest()
    idx = int(file_hash, 16) % len(scenes)
    return scenes[idx]

def label_speech_files(seed=1337):
    """Main scene labeling function"""
    print("=" * 70)
    print("SceneGuard T40: Scene Labeling")
    print("=" * 70)
    
    # Setup
    speech_dir = Path("data/raw/speech/libritts/5339")
    output_file = Path("data/interim/scene_labels.csv")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[1/4] Scanning speech files...")
    print(f"  Directory: {speech_dir}")
    
    # Get all wav files
    wav_files = sorted(speech_dir.glob("*.wav"))
    print(f"  Found: {len(wav_files)} audio files")
    
    if len(wav_files) == 0:
        print("  ✗ No audio files found!")
        return 1
    
    # Assign scenes
    print(f"\n[2/4] Assigning scene labels...")
    print(f"  Method: Heuristic (hash-based)")
    print(f"  Seed: {seed}")
    print(f"  Available scenes: {len(TAU_SCENES)}")
    
    results = []
    scene_counts = {scene: 0 for scene in TAU_SCENES}
    
    for audio_path in tqdm(wav_files, desc="  Labeling"):
        scene = assign_scene_heuristic(audio_path, TAU_SCENES, seed)
        scene_counts[scene] += 1
        
        # Handle both absolute and relative paths
        if audio_path.is_absolute():
            rel_path = str(audio_path.relative_to(Path.cwd()))
        else:
            rel_path = str(audio_path)
        
        results.append({
            'audio_path': rel_path,
            'scene_label': scene,
            'confidence': 1.0,  # Heuristic = 100% confidence
            'method': 'heuristic_hash',
            'seed': seed
        })
    
    # Create DataFrame
    print(f"\n[3/4] Scene distribution:")
    df = pd.DataFrame(results)
    for scene, count in sorted(scene_counts.items()):
        percentage = 100 * count / len(wav_files)
        print(f"  {scene:20s}: {count:3d} files ({percentage:5.1f}%)")
    
    # Save
    print(f"\n[4/4] Saving results...")
    df.to_csv(output_file, index=False)
    print(f"  Saved to: {output_file}")
    print(f"  Total rows: {len(df)}")
    
    # Validation
    print("\n" + "=" * 70)
    print("Validation")
    print("=" * 70)
    
    checks = [
        ("Coverage", len(df) == len(wav_files), f"{len(df)}/{len(wav_files)} files labeled"),
        ("No missing scenes", all(count > 0 for count in scene_counts.values()), "All scenes represented"),
        ("No nulls", df.isnull().sum().sum() == 0, "No missing values"),
        ("Valid scenes", df['scene_label'].isin(TAU_SCENES).all(), "All scenes valid"),
    ]
    
    all_pass = True
    for check_name, passed, message in checks:
        status = "✓" if passed else "✗"
        print(f"  {status} {check_name}: {message}")
        if not passed:
            all_pass = False
    
    if all_pass:
        print("\n✓ Scene labeling complete and validated!")
        return 0
    else:
        print("\n✗ Validation failed")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(label_speech_files(seed=1337))

