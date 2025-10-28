"""
Loss Functions for SceneGuard Optimization

Implements:
- L_SIM: Speaker similarity loss (minimize)
- L_ASR: ASR usability loss (optional)
- L_SCN: Scene consistency loss (optional)
- L_REG: Regularization loss (mask smoothness + energy)
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional


def compute_speaker_similarity_loss(
    protected_audio: torch.Tensor,
    clean_embedding: torch.Tensor,
    speaker_encoder
) -> torch.Tensor:
    """
    Compute speaker similarity loss: cosine similarity between embeddings
    
    Goal: Minimize similarity to degrade speaker identity
    
    Args:
        protected_audio: Protected speech tensor [B, T] or [1, T]
        clean_embedding: Clean speech embedding from speaker encoder [D]
        speaker_encoder: Pre-trained speaker verification model
    
    Returns:
        loss: Cosine similarity (to minimize)
    """
    # Ensure 2D input: [B, T]
    if protected_audio.dim() > 2:
        protected_audio = protected_audio.squeeze(1)  # Remove channel dim if present
    elif protected_audio.dim() == 1:
        protected_audio = protected_audio.unsqueeze(0)  # Add batch dim
    
    # Extract embedding from protected audio
    with torch.set_grad_enabled(True):
        protected_embedding = speaker_encoder.encode_batch(protected_audio).squeeze()
    
    # Compute cosine similarity
    similarity = F.cosine_similarity(
        clean_embedding.unsqueeze(0),
        protected_embedding.unsqueeze(0),
        dim=1
    )
    
    # Return similarity as loss (we want to minimize it)
    return similarity.mean()


def compute_asr_loss(
    protected_audio: np.ndarray,
    reference_text: str,
    asr_model,
    sr: int = 16000
) -> torch.Tensor:
    """
    Compute ASR loss: penalize transcription errors
    
    Goal: Maintain intelligibility (minimize WER)
    
    Note: This is computationally expensive and typically disabled
    during optimization. Used mainly for evaluation.
    
    Args:
        protected_audio: Protected speech numpy array [T]
        reference_text: Ground truth transcription
        asr_model: ASR model (e.g., Whisper)
        sr: Sampling rate
    
    Returns:
        loss: ASR-based loss (placeholder, requires implementation)
    """
    # Placeholder implementation
    # In practice, this would:
    # 1. Transcribe protected_audio using asr_model
    # 2. Compute CTC loss or word error rate
    # 3. Return differentiable loss
    
    # For now, return zero (ASR loss disabled by default)
    return torch.tensor(0.0)


def compute_scene_consistency_loss(
    protected_audio: np.ndarray,
    scene_label: str,
    asc_model,
    sr: int = 16000
) -> torch.Tensor:
    """
    Compute scene consistency loss: ensure protected audio maintains scene label
    
    Goal: Preserve acoustic scene consistency
    
    Args:
        protected_audio: Protected speech numpy array [T]
        scene_label: Target scene label
        asc_model: Acoustic scene classification model
        sr: Sampling rate
    
    Returns:
        loss: Negative log confidence of target scene
    """
    # Placeholder implementation
    # In practice, this would:
    # 1. Extract features from protected_audio
    # 2. Get scene probabilities from asc_model
    # 3. Return -log(P(scene_label))
    
    # For now, return zero (scene loss disabled by default)
    return torch.tensor(0.0)


def compute_regularization_loss(
    mask: torch.Tensor,
    gamma: torch.Tensor,
    smooth_weight: float = 1.0,
    energy_weight: float = 0.1
) -> torch.Tensor:
    """
    Compute regularization loss: encourage smooth masks and bounded gamma
    
    Goal: Prevent extreme/spiky solutions
    
    Args:
        mask: Temporal mask [T]
        gamma: Noise strength scalar
        smooth_weight: Weight for smoothness regularization
        energy_weight: Weight for energy regularization
    
    Returns:
        loss: Combined regularization loss
    """
    # Smoothness: penalize large gradients (TV loss)
    # Compute finite differences
    mask_diff = mask[1:] - mask[:-1]
    smooth_loss = torch.mean(mask_diff ** 2)
    
    # Energy: penalize large gamma
    energy_loss = gamma ** 2
    
    # Combined regularization
    reg_loss = smooth_weight * smooth_loss + energy_weight * energy_loss
    
    return reg_loss


def compute_total_loss(
    protected_audio: torch.Tensor,
    clean_embedding: torch.Tensor,
    speaker_encoder,
    mask: torch.Tensor,
    gamma: torch.Tensor,
    lambda_sim: float = 1.0,
    lambda_reg: float = 0.01,
    lambda_asr: float = 0.0,
    lambda_scn: float = 0.0,
    asr_model=None,
    asc_model=None,
    reference_text: Optional[str] = None,
    scene_label: Optional[str] = None,
    sr: int = 16000
) -> torch.Tensor:
    """
    Compute total optimization loss
    
    L = λ_SIM * L_SIM + λ_REG * L_REG + λ_ASR * L_ASR + λ_SCN * L_SCN
    
    Args:
        protected_audio: Protected speech tensor [B, 1, T]
        clean_embedding: Clean speech embedding [D]
        speaker_encoder: Speaker verification model
        mask: Temporal mask [T]
        gamma: Noise strength scalar
        lambda_sim: Weight for speaker similarity loss
        lambda_reg: Weight for regularization loss
        lambda_asr: Weight for ASR loss
        lambda_scn: Weight for scene consistency loss
        asr_model: Optional ASR model
        asc_model: Optional ASC model
        reference_text: Optional reference text
        scene_label: Optional scene label
        sr: Sampling rate
    
    Returns:
        total_loss: Weighted sum of all losses
    """
    # Speaker similarity loss
    loss_sim = compute_speaker_similarity_loss(
        protected_audio, clean_embedding, speaker_encoder
    )
    
    # Regularization loss
    loss_reg = compute_regularization_loss(mask, gamma)
    
    # Start with main losses
    total_loss = lambda_sim * loss_sim + lambda_reg * loss_reg
    
    # Optional ASR loss
    if lambda_asr > 0 and asr_model is not None and reference_text is not None:
        protected_np = protected_audio.squeeze().cpu().detach().numpy()
        loss_asr = compute_asr_loss(protected_np, reference_text, asr_model, sr)
        total_loss += lambda_asr * loss_asr
    
    # Optional scene consistency loss
    if lambda_scn > 0 and asc_model is not None and scene_label is not None:
        protected_np = protected_audio.squeeze().cpu().detach().numpy()
        loss_scn = compute_scene_consistency_loss(protected_np, scene_label, asc_model, sr)
        total_loss += lambda_scn * loss_scn
    
    return total_loss

