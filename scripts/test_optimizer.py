#!/usr/bin/env python3
"""
Test SceneGuard Optimizer on a single sample

Quick validation that optimization framework works correctly.
"""

import sys
from pathlib import Path
import torch
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.sceneguard.mixing.mixer import SceneGuardMixer
from speechbrain.pretrained import EncoderClassifier


def test_optimizer():
    """Test optimizer on one sample"""
    print("=" * 70)
    print("Testing SceneGuard Optimizer")
    print("=" * 70)
    
    # Check CUDA
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    # Load speaker encoder
    print("\n[1/4] Loading ECAPA-TDNN speaker encoder...")
    speaker_encoder = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="models/sv/ecapa/",
        run_opts={"device": device}
    )
    print("  ✓ Loaded")
    
    # Load scene labels
    print("\n[2/4] Loading scene labels...")
    scene_labels_path = project_root / "data/interim/scene_labels.csv"
    scene_df = pd.read_csv(scene_labels_path)
    print(f"  ✓ Loaded {len(scene_df)} scene labels")
    
    # Select one test sample
    test_sample = scene_df.iloc[0]
    speech_path = project_root / test_sample['audio_path']
    scene_label = test_sample['scene_label']
    
    print(f"\n[3/4] Test sample:")
    print(f"  Audio: {speech_path.name}")
    print(f"  Scene: {scene_label}")
    
    # Create mixer
    mixer = SceneGuardMixer(
        noise_taxonomy_path=str(project_root / "artifacts/noise_taxonomy.json"),
        snr_range=(10.0, 20.0),
        masking_strategy="stochastic",
        seed=1337
    )
    
    # Test optimization (reduced epochs for speed)
    print("\n[4/4] Running optimization (20 epochs for testing)...")
    output_path = project_root / "artifacts/test_optimizer_output.wav"
    log_path = project_root / "artifacts/test_optimizer_log.json"
    
    try:
        mixed, params = mixer.mix_optimized(
            speech_path=str(speech_path),
            scene=scene_label,
            speaker_encoder=speaker_encoder,
            output_path=str(output_path),
            max_epochs=20,  # Reduced for quick test
            lambda_sim=1.0,
            lambda_reg=0.01,
            lr=0.01,
            log_path=str(log_path)
        )
        
        print("\n" + "=" * 70)
        print("✓ Optimization Complete!")
        print("=" * 70)
        print(f"\nFinal Results:")
        print(f"  SNR: {params['final_snr_db']:.2f} dB")
        print(f"  γ: {params['final_gamma']:.4f}")
        print(f"  Similarity: {params['final_similarity']:.4f}")
        print(f"  Mask mean: {params['mask_mean']:.4f}")
        print(f"  Mask std: {params['mask_std']:.4f}")
        print(f"\nOutput saved to: {output_path}")
        print(f"Log saved to: {log_path}")
        
        print("\n✓ Test PASSED - Optimizer is working correctly!")
        return 0
        
    except Exception as e:
        print(f"\n✗ Test FAILED with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(test_optimizer())

