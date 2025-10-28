"""
SceneGuard Optimizer

Implements gradient-based optimization for scene-consistent noise mixing.
Jointly optimizes temporal mask m(t) and noise strength γ.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Dict, Optional, Tuple
import json

from .losses import (
    compute_speaker_similarity_loss,
    compute_asr_loss,
    compute_scene_consistency_loss,
    compute_regularization_loss,
)


class SceneGuardOptimizer:
    """
    SceneGuard Optimization Framework
    
    Jointly optimizes temporal mask m(t) and noise strength γ to:
    1. Minimize speaker similarity (L_SIM)
    2. Maintain usability (L_ASR, optional)
    3. Preserve scene consistency (L_SCN)
    4. Regularize parameters (L_REG)
    
    Args:
        speaker_encoder: Pre-trained speaker verification model (ECAPA-TDNN)
        asr_model: Optional ASR model for usability constraint (Whisper)
        asc_model: Optional acoustic scene classifier
        snr_range: (min_snr, max_snr) in dB, default (10, 20)
        lambda_sim: Weight for speaker similarity loss
        lambda_asr: Weight for ASR loss (0 to disable)
        lambda_scn: Weight for scene consistency loss (0 to disable)
        lambda_reg: Weight for regularization
        max_epochs: Maximum optimization iterations
        lr: Learning rate for Adam optimizer
        device: 'cuda' or 'cpu'
        seed: Random seed
    """
    
    def __init__(
        self,
        speaker_encoder,
        asr_model=None,
        asc_model=None,
        snr_range: Tuple[float, float] = (10.0, 20.0),
        lambda_sim: float = 1.0,
        lambda_asr: float = 0.0,
        lambda_scn: float = 0.0,
        lambda_reg: float = 0.01,
        max_epochs: int = 50,
        lr: float = 0.01,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        seed: int = 1337
    ):
        self.speaker_encoder = speaker_encoder.to(device).eval()
        self.asr_model = asr_model
        self.asc_model = asc_model
        
        self.snr_range = snr_range
        self.lambda_sim = lambda_sim
        self.lambda_asr = lambda_asr
        self.lambda_scn = lambda_scn
        self.lambda_reg = lambda_reg
        self.max_epochs = max_epochs
        self.lr = lr
        self.device = device
        self.seed = seed
        
        # Compute SNR to gamma mapping parameters
        self.snr_min, self.snr_max = snr_range
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        print(f"SceneGuard Optimizer initialized:")
        print(f"  Device: {device}")
        print(f"  SNR range: [{self.snr_min}, {self.snr_max}] dB")
        print(f"  λ_SIM={lambda_sim}, λ_ASR={lambda_asr}, λ_SCN={lambda_scn}, λ_REG={lambda_reg}")
        print(f"  Max epochs: {max_epochs}, LR: {lr}")
    
    def project_gamma_to_snr(self, gamma_raw: torch.Tensor, 
                             speech_power: float, noise_power: float) -> torch.Tensor:
        """
        Project raw gamma parameter to valid SNR range [snr_min, snr_max] dB
        
        SNR(dB) = 10 * log10(P_speech / P_noise)
        where P_noise = gamma^2 * noise_power
        
        Args:
            gamma_raw: Unconstrained parameter
            speech_power: Power of speech signal
            noise_power: Power of noise signal
        
        Returns:
            gamma: Constrained gamma value corresponding to SNR in [snr_min, snr_max]
        """
        # Map gamma_raw through sigmoid to [0, 1]
        alpha = torch.sigmoid(gamma_raw)
        
        # Compute gamma values corresponding to SNR bounds
        # SNR = 10*log10(P_speech / (gamma^2 * P_noise))
        # gamma = sqrt(P_speech / (P_noise * 10^(SNR/10)))
        
        if noise_power < 1e-10:
            return torch.tensor(0.0, device=self.device)
        
        snr_min_linear = 10 ** (self.snr_min / 10.0)
        snr_max_linear = 10 ** (self.snr_max / 10.0)
        
        gamma_max = np.sqrt(speech_power / (noise_power * snr_min_linear))  # Higher gamma = lower SNR
        gamma_min = np.sqrt(speech_power / (noise_power * snr_max_linear))  # Lower gamma = higher SNR
        
        # Linearly interpolate between gamma_min and gamma_max
        gamma = gamma_min + alpha * (gamma_max - gamma_min)
        
        return gamma
    
    def get_mask(self, mask_logits: torch.Tensor) -> torch.Tensor:
        """
        Project mask logits to [0, 1] range using sigmoid
        
        Args:
            mask_logits: Unconstrained mask parameters
        
        Returns:
            mask: Temporal mask in [0, 1]
        """
        return torch.sigmoid(mask_logits)
    
    def optimize(
        self,
        speech: np.ndarray,
        noise: np.ndarray,
        scene_label: Optional[str] = None,
        reference_text: Optional[str] = None,
        sr: int = 16000,
        save_trajectory: bool = True
    ) -> Tuple[np.ndarray, Dict]:
        """
        Optimize temporal mask and noise strength for given speech
        
        Args:
            speech: Clean speech signal [T]
            noise: Scene-consistent noise signal [T] (must match speech length)
            scene_label: Optional scene label for consistency loss
            reference_text: Optional reference text for ASR loss
            sr: Sampling rate
            save_trajectory: Whether to save optimization trajectory
        
        Returns:
            protected_speech: Optimized protected speech
            result_dict: Dictionary containing final parameters and losses
        """
        T = len(speech)
        assert len(noise) == T, f"Speech and noise length mismatch: {T} vs {len(noise)}"
        
        # Convert to torch tensors
        speech_tensor = torch.from_numpy(speech).float().to(self.device)
        noise_tensor = torch.from_numpy(noise).float().to(self.device)
        
        # Compute powers for SNR constraint
        speech_power = torch.mean(speech_tensor ** 2).item()
        noise_power = torch.mean(noise_tensor ** 2).item()
        
        # Initialize optimizable parameters
        mask_logits = nn.Parameter(torch.randn(T, device=self.device) * 0.1)
        gamma_raw = nn.Parameter(torch.tensor(0.0, device=self.device))
        
        # Setup optimizer
        optimizer = optim.Adam([mask_logits, gamma_raw], lr=self.lr)
        
        # Optimization trajectory
        trajectory = {
            'losses': [],
            'sim_losses': [],
            'reg_losses': [],
            'gammas': [],
            'snrs': [],
        }
        
        # Extract clean speaker embedding (target to minimize similarity)
        with torch.no_grad():
            clean_embedding = self.speaker_encoder.encode_batch(
                speech_tensor.unsqueeze(0)  # [1, T]
            ).squeeze()
        
        # Optimization loop
        for epoch in range(self.max_epochs):
            optimizer.zero_grad()
            
            # Project parameters
            mask = self.get_mask(mask_logits)
            gamma = self.project_gamma_to_snr(gamma_raw, speech_power, noise_power)
            
            # Forward: Mix speech with noise
            protected_tensor = speech_tensor + gamma * mask * noise_tensor
            
            # Normalize to prevent clipping
            max_val = torch.max(torch.abs(protected_tensor))
            if max_val > 0.99:
                protected_tensor = protected_tensor / max_val * 0.99
            
            # Compute speaker similarity loss
            loss_sim = compute_speaker_similarity_loss(
                protected_tensor.unsqueeze(0),  # [1, T]
                clean_embedding,
                self.speaker_encoder
            )
            
            # Compute regularization loss
            loss_reg = compute_regularization_loss(mask, gamma)
            
            # Total loss
            total_loss = (
                self.lambda_sim * loss_sim +
                self.lambda_reg * loss_reg
            )
            
            # Optional: ASR loss (disabled by default due to computation cost)
            if self.lambda_asr > 0 and self.asr_model is not None and reference_text is not None:
                loss_asr = compute_asr_loss(
                    protected_tensor.cpu().numpy(),
                    reference_text,
                    self.asr_model,
                    sr
                )
                total_loss += self.lambda_asr * loss_asr
            
            # Optional: Scene consistency loss
            if self.lambda_scn > 0 and self.asc_model is not None and scene_label is not None:
                loss_scn = compute_scene_consistency_loss(
                    protected_tensor.cpu().numpy(),
                    scene_label,
                    self.asc_model,
                    sr
                )
                total_loss += self.lambda_scn * loss_scn
            
            # Backward
            total_loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_([mask_logits, gamma_raw], max_norm=1.0)
            
            optimizer.step()
            
            # Log trajectory
            if save_trajectory:
                # Compute actual SNR
                protected_power = torch.mean(protected_tensor ** 2).item()
                noise_contribution_power = torch.mean((gamma * mask * noise_tensor) ** 2).item()
                actual_snr = 10 * np.log10(speech_power / (noise_contribution_power + 1e-10))
                
                trajectory['losses'].append(total_loss.item())
                trajectory['sim_losses'].append(loss_sim.item())
                trajectory['reg_losses'].append(loss_reg.item())
                trajectory['gammas'].append(gamma.item())
                trajectory['snrs'].append(actual_snr)
            
            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1}/{self.max_epochs}: "
                      f"Loss={total_loss.item():.4f}, "
                      f"SIM={loss_sim.item():.4f}, "
                      f"γ={gamma.item():.4f}, "
                      f"SNR={actual_snr:.2f}dB")
        
        # Final protected speech
        with torch.no_grad():
            final_mask = self.get_mask(mask_logits)
            final_gamma = self.project_gamma_to_snr(gamma_raw, speech_power, noise_power)
            final_protected = speech_tensor + final_gamma * final_mask * noise_tensor
            
            # Normalize
            max_val = torch.max(torch.abs(final_protected))
            if max_val > 0.99:
                final_protected = final_protected / max_val * 0.99
            
            # Compute final metrics
            final_embedding = self.speaker_encoder.encode_batch(
                final_protected.unsqueeze(0)  # [1, T]
            ).squeeze()
            
            final_sim = torch.nn.functional.cosine_similarity(
                clean_embedding.unsqueeze(0),
                final_embedding.unsqueeze(0)
            ).item()
            
            final_noise_power = torch.mean((final_gamma * final_mask * noise_tensor) ** 2).item()
            final_snr = 10 * np.log10(speech_power / (final_noise_power + 1e-10))
        
        # Convert to numpy
        protected_speech = final_protected.cpu().numpy()
        
        # Prepare result dictionary
        result_dict = {
            'final_gamma': final_gamma.item(),
            'final_snr': final_snr,
            'final_similarity': final_sim,
            'mask_mean': final_mask.mean().item(),
            'mask_std': final_mask.std().item(),
            'speech_rms': np.sqrt(speech_power),
            'noise_rms': np.sqrt(noise_power),
            'protected_rms': np.sqrt(np.mean(protected_speech ** 2)),
            'trajectory': trajectory if save_trajectory else None,
        }
        
        return protected_speech, result_dict
    
    def optimize_and_save(
        self,
        speech_path: str,
        noise_path: str,
        output_path: str,
        scene_label: Optional[str] = None,
        reference_text: Optional[str] = None,
        sr: int = 16000,
        log_path: Optional[str] = None
    ) -> Dict:
        """
        Convenience function: load audio, optimize, and save results
        
        Args:
            speech_path: Path to clean speech
            noise_path: Path to scene-consistent noise
            output_path: Path to save protected speech
            scene_label: Optional scene label
            reference_text: Optional reference text
            sr: Sampling rate
            log_path: Optional path to save optimization log (JSON)
        
        Returns:
            result_dict: Dictionary with final parameters and losses
        """
        # Load audio
        speech, _ = librosa.load(speech_path, sr=sr, mono=True)
        noise, _ = librosa.load(noise_path, sr=sr, mono=True)
        
        # Match lengths
        if len(noise) < len(speech):
            # Loop noise
            num_repeats = int(np.ceil(len(speech) / len(noise)))
            noise = np.tile(noise, num_repeats)
        noise = noise[:len(speech)]
        
        # Optimize
        protected_speech, result_dict = self.optimize(
            speech, noise, scene_label, reference_text, sr
        )
        
        # Save protected audio
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, protected_speech, sr)
        
        # Save log if requested
        if log_path:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Convert numpy types to native Python for JSON serialization
            log_dict = {
                'speech_path': speech_path,
                'noise_path': noise_path,
                'output_path': output_path,
                'scene_label': scene_label,
                'final_gamma': float(result_dict['final_gamma']),
                'final_snr': float(result_dict['final_snr']),
                'final_similarity': float(result_dict['final_similarity']),
                'mask_mean': float(result_dict['mask_mean']),
                'mask_std': float(result_dict['mask_std']),
                'speech_rms': float(result_dict['speech_rms']),
                'noise_rms': float(result_dict['noise_rms']),
                'protected_rms': float(result_dict['protected_rms']),
            }
            
            # Save trajectory separately if present
            if result_dict['trajectory']:
                log_dict['trajectory'] = {
                    'losses': [float(x) for x in result_dict['trajectory']['losses']],
                    'sim_losses': [float(x) for x in result_dict['trajectory']['sim_losses']],
                    'reg_losses': [float(x) for x in result_dict['trajectory']['reg_losses']],
                    'gammas': [float(x) for x in result_dict['trajectory']['gammas']],
                    'snrs': [float(x) for x in result_dict['trajectory']['snrs']],
                }
            
            with open(log_path, 'w') as f:
                json.dump(log_dict, f, indent=2)
        
        return result_dict

