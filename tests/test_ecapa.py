#!/usr/bin/env python3
"""
Test ECAPA-TDNN speaker verification model
"""
import torch
import torchaudio
from pathlib import Path

def test_ecapa():
    print("=" * 60)
    print("Testing ECAPA-TDNN")
    print("=" * 60)
    
    try:
        from speechbrain.pretrained import EncoderClassifier
        
        # Load model
        print("Loading ECAPA-TDNN...")
        model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="models/sv/ecapa/"
        )
        print("✓ Model loaded")
        
        # Test with dummy audio
        print("\nTesting with dummy audio (1 second @ 16kHz)...")
        dummy_wav = torch.randn(1, 16000)
        
        with torch.no_grad():
            embedding = model.encode_batch(dummy_wav)
        
        print(f"✓ Embedding shape: {embedding.shape}")
        print(f"  Expected: [1, D] where D is embedding dimension")
        print(f"  Actual dimension: {embedding.shape[-1]}")
        
        # Verify dimension
        assert len(embedding.shape) == 2, "Embedding should be 2D"
        assert embedding.shape[0] == 1, "Batch size should be 1"
        assert embedding.shape[-1] in [192, 256, 512], f"Unexpected embedding dimension: {embedding.shape[-1]}"
        
        # Test with real audio if available
        test_audio = Path("data/raw/speech/libritts/5339")
        if test_audio.exists():
            wav_files = list(test_audio.glob("*.wav"))
            if wav_files:
                print(f"\nTesting with real audio: {wav_files[0].name}")
                wav, sr = torchaudio.load(wav_files[0])
                if sr != 16000:
                    wav = torchaudio.functional.resample(wav, sr, 16000)
                
                with torch.no_grad():
                    real_emb = model.encode_batch(wav)
                
                print(f"✓ Real audio embedding shape: {real_emb.shape}")
                print(f"  Norm: {torch.norm(real_emb).item():.4f}")
        
        print("\n" + "=" * 60)
        print("✓ ECAPA-TDNN test PASSED")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n✗ ECAPA-TDNN test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import sys
    success = test_ecapa()
    sys.exit(0 if success else 1)

