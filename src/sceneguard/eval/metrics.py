"""
SceneGuard Metrics Computation
Implements: SIM, WER, PESQ, STOI, MCD
"""
import torch
import numpy as np
import librosa
from pathlib import Path
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

def load_ecapa_model():
    """Load ECAPA-TDNN speaker verification model"""
    try:
        from speechbrain.inference import EncoderClassifier
        model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="models/sv/ecapa/"
        )
        return model
    except Exception as e:
        print(f"Warning: Could not load ECAPA model: {e}")
        return None

def compute_speaker_similarity(audio1_path: str, audio2_path: str, model=None) -> float:
    """
    Compute speaker similarity using ECAPA-TDNN
    
    Args:
        audio1_path: Path to first audio
        audio2_path: Path to second audio
        model: Optional pre-loaded ECAPA model
    
    Returns:
        similarity: Cosine similarity score [-1, 1]
    """
    if model is None:
        model = load_ecapa_model()
    
    if model is None:
        return float('nan')
    
    try:
        import torchaudio
        
        # Load audios
        wav1, sr1 = torchaudio.load(audio1_path)
        wav2, sr2 = torchaudio.load(audio2_path)
        
        # Resample if needed
        if sr1 != 16000:
            wav1 = torchaudio.functional.resample(wav1, sr1, 16000)
        if sr2 != 16000:
            wav2 = torchaudio.functional.resample(wav2, sr2, 16000)
        
        # Extract embeddings
        with torch.no_grad():
            emb1 = model.encode_batch(wav1)
            emb2 = model.encode_batch(wav2)
        
        # Flatten embeddings to 1D if needed
        emb1 = emb1.squeeze()
        emb2 = emb2.squeeze()
        
        # Ensure they're 1D
        if len(emb1.shape) > 1:
            emb1 = emb1.flatten()
        if len(emb2.shape) > 1:
            emb2 = emb2.flatten()
        
        # Compute cosine similarity
        similarity = torch.nn.functional.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
        
        return float(similarity)
    
    except Exception as e:
        print(f"Error computing similarity: {e}")
        return float('nan')

def compute_wer(reference_path: str, hypothesis_path: str, model=None) -> float:
    """
    Compute Word Error Rate using Whisper
    
    Args:
        reference_path: Path to reference audio
        hypothesis_path: Path to hypothesis audio
        model: Optional pre-loaded Whisper model
    
    Returns:
        wer: Word error rate [0, ∞]
    """
    if model is None:
        import whisper
        model = whisper.load_model("base", download_root="models/asr/whisper/")
    
    try:
        # Transcribe both
        ref_result = model.transcribe(reference_path, fp16=False, language="en")
        hyp_result = model.transcribe(hypothesis_path, fp16=False, language="en")
        
        ref_text = ref_result['text'].strip()
        hyp_text = hyp_result['text'].strip()
        
        # Compute WER
        from jiwer import wer as compute_wer_jiwer
        error_rate = compute_wer_jiwer(ref_text, hyp_text)
        
        return float(error_rate)
    
    except Exception as e:
        print(f"Error computing WER: {e}")
        return float('nan')

def compute_pesq(reference_path: str, degraded_path: str, sr: int = 16000) -> float:
    """
    Compute PESQ (Perceptual Evaluation of Speech Quality)
    
    Args:
        reference_path: Path to reference audio
        degraded_path: Path to degraded audio
        sr: Sampling rate (8000 or 16000)
    
    Returns:
        pesq_score: PESQ score [1, 4.5]
    """
    try:
        from pesq import pesq as compute_pesq_lib
        import soundfile as sf
        
        # Load audio
        ref, ref_sr = sf.read(reference_path)
        deg, deg_sr = sf.read(degraded_path)
        
        # Resample if needed
        if ref_sr != sr:
            ref = librosa.resample(ref, orig_sr=ref_sr, target_sr=sr)
        if deg_sr != sr:
            deg = librosa.resample(deg, orig_sr=deg_sr, target_sr=sr)
        
        # Match lengths
        min_len = min(len(ref), len(deg))
        ref = ref[:min_len]
        deg = deg[:min_len]
        
        # Compute PESQ
        score = compute_pesq_lib(sr, ref, deg, 'wb' if sr == 16000 else 'nb')
        
        return float(score)
    
    except Exception as e:
        print(f"Error computing PESQ: {e}")
        return float('nan')

def compute_stoi(reference_path: str, degraded_path: str, sr: int = 16000) -> float:
    """
    Compute STOI (Short-Time Objective Intelligibility)
    
    Args:
        reference_path: Path to reference audio
        degraded_path: Path to degraded audio
        sr: Sampling rate
    
    Returns:
        stoi_score: STOI score [0, 1]
    """
    try:
        from pystoi import stoi as compute_stoi_lib
        import soundfile as sf
        
        # Load audio
        ref, ref_sr = sf.read(reference_path)
        deg, deg_sr = sf.read(degraded_path)
        
        # Resample if needed
        if ref_sr != sr:
            ref = librosa.resample(ref, orig_sr=ref_sr, target_sr=sr)
        if deg_sr != sr:
            deg = librosa.resample(deg, orig_sr=deg_sr, target_sr=sr)
        
        # Match lengths
        min_len = min(len(ref), len(deg))
        ref = ref[:min_len]
        deg = deg[:min_len]
        
        # Compute STOI
        score = compute_stoi_lib(ref, deg, sr, extended=False)
        
        return float(score)
    
    except Exception as e:
        print(f"Error computing STOI: {e}")
        return float('nan')

def compute_mcd(reference_path: str, synthesized_path: str) -> float:
    """
    Compute Mel-Cepstral Distortion
    
    Args:
        reference_path: Path to reference audio
        synthesized_path: Path to synthesized audio
    
    Returns:
        mcd: MCD in dB
    """
    try:
        import soundfile as sf
        
        # Load audio
        ref, ref_sr = sf.read(reference_path)
        syn, syn_sr = sf.read(synthesized_path)
        
        # Resample to 16kHz
        if ref_sr != 16000:
            ref = librosa.resample(ref, orig_sr=ref_sr, target_sr=16000)
        if syn_sr != 16000:
            syn = librosa.resample(syn, orig_sr=syn_sr, target_sr=16000)
        
        # Extract mel-cepstral coefficients
        ref_mfcc = librosa.feature.mfcc(y=ref, sr=16000, n_mfcc=13)
        syn_mfcc = librosa.feature.mfcc(y=syn, sr=16000, n_mfcc=13)
        
        # Match lengths
        min_frames = min(ref_mfcc.shape[1], syn_mfcc.shape[1])
        ref_mfcc = ref_mfcc[:, :min_frames]
        syn_mfcc = syn_mfcc[:, :min_frames]
        
        # Compute MCD
        diff = ref_mfcc - syn_mfcc
        mcd = np.mean(np.sqrt(np.sum(diff**2, axis=0))) * (10.0 / np.log(10)) * np.sqrt(2)
        
        return float(mcd)
    
    except Exception as e:
        print(f"Error computing MCD: {e}")
        return float('nan')

def compute_all_metrics(clean_path: str, defended_path: str) -> Dict[str, float]:
    """
    Compute all metrics comparing clean and defended audio
    
    Args:
        clean_path: Path to clean audio
        defended_path: Path to defended audio
    
    Returns:
        metrics: Dictionary of all metrics
    """
    metrics = {}
    
    # Speaker similarity (how similar defended is to clean)
    metrics['similarity'] = compute_speaker_similarity(clean_path, defended_path)
    
    # WER (transcription accuracy)
    metrics['wer'] = compute_wer(clean_path, defended_path)
    
    # Quality metrics
    metrics['pesq'] = compute_pesq(clean_path, defended_path)
    metrics['stoi'] = compute_stoi(clean_path, defended_path)
    metrics['mcd'] = compute_mcd(clean_path, defended_path)
    
    return metrics

