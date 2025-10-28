#!/usr/bin/env python3
"""
Test Whisper ASR model
"""
import torch
import whisper
import torchaudio
from pathlib import Path

def test_whisper():
    print("=" * 60)
    print("Testing Whisper ASR")
    print("=" * 60)
    
    try:
        # Load model
        print("Loading Whisper base model...")
        model = whisper.load_model("base", download_root="models/asr/whisper/")
        print("✓ Model loaded")
        
        # Test with real audio if available
        test_audio = Path("data/raw/speech/libritts/5339")
        if test_audio.exists():
            wav_files = list(test_audio.glob("*.wav"))
            if wav_files:
                test_file = wav_files[0]
                print(f"\nTranscribing: {test_file.name}")
                
                # Load and transcribe
                result = model.transcribe(str(test_file), language="en", fp16=False)
                
                print(f"✓ Transcription: '{result['text']}'")
                print(f"  Language: {result.get('language', 'N/A')}")
                
                # Test WER calculation
                from jiwer import wer
                ref = "this is a test"
                hyp = "this is a test"
                error_rate = wer(ref, hyp)
                print(f"\n✓ WER calculation test: {error_rate:.2%} (expected 0%)")
                assert error_rate == 0.0, "WER should be 0 for identical strings"
        else:
            print("\n⚠️  No test audio found, testing with dummy audio...")
            # Whisper requires 30-second audio at 16kHz
            dummy_audio = torch.zeros(480000).numpy()  # 30 sec silence
            result = model.transcribe(dummy_audio, fp16=False)
            print(f"  Dummy transcription: '{result['text']}'")
        
        print("\n" + "=" * 60)
        print("✓ Whisper ASR test PASSED")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n✗ Whisper ASR test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import sys
    success = test_whisper()
    sys.exit(0 if success else 1)

