"""
SceneGuard Mixer: Core defense algorithm
Implements: x'(t) = x(t) + γ·n_scene(t)·m(t)
"""
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import random
import json
from typing import Tuple, Optional, Dict

class SceneGuardMixer:
    """
    SceneGuard Defense Mixer
    
    Mixes scene-consistent background noise with clean speech to protect
    against voice cloning attacks.
    
    Args:
        noise_taxonomy_path: Path to noise_taxonomy.json
        snr_range: (min_snr, max_snr) in dB, e.g. (10, 20)
        masking_strategy: 'stochastic', 'gaps', or 'ramp'
        target_sr: Target sampling rate (Hz)
        seed: Random seed for reproducibility
    """
    
    def __init__(
        self,
        noise_taxonomy_path: str = "artifacts/noise_taxonomy.json",
        snr_range: Tuple[float, float] = (10.0, 20.0),
        masking_strategy: str = "stochastic",
        target_sr: int = 16000,
        seed: int = 1337
    ):
        self.snr_range = snr_range
        self.masking_strategy = masking_strategy
        self.target_sr = target_sr
        self.seed = seed
        
        # Load noise taxonomy
        with open(noise_taxonomy_path, 'r') as f:
            self.noise_taxonomy = json.load(f)
        
        print(f"SceneGuard Mixer initialized:")
        print(f"  SNR range: {snr_range[0]}-{snr_range[1]} dB")
        print(f"  Masking: {masking_strategy}")
        print(f"  Scenes: {len(self.noise_taxonomy)}")
        print(f"  Total noise clips: {sum(len(v) for v in self.noise_taxonomy.values())}")
    
    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Load audio and resample to target SR"""
        audio, sr = librosa.load(audio_path, sr=self.target_sr, mono=True)
        return audio, sr
    
    def sample_noise(self, scene: str, duration: float) -> np.ndarray:
        """
        Sample noise clip from scene and loop/trim to match duration
        
        Args:
            scene: Scene label (e.g., 'park', 'street_traffic')
            duration: Target duration in seconds
        
        Returns:
            noise: Noise array matching duration
        """
        if scene not in self.noise_taxonomy:
            raise ValueError(f"Scene '{scene}' not in taxonomy")
        
        noise_clips = self.noise_taxonomy[scene]
        if len(noise_clips) == 0:
            raise ValueError(f"No noise clips available for scene '{scene}'")
        
        # Randomly select a noise clip
        noise_path = random.choice(noise_clips)
        noise, _ = self.load_audio(noise_path)
        
        # Match duration by looping or trimming
        target_samples = int(duration * self.target_sr)
        
        if len(noise) < target_samples:
            # Loop noise to match duration
            num_repeats = int(np.ceil(target_samples / len(noise)))
            noise = np.tile(noise, num_repeats)
        
        # Trim to exact duration
        noise = noise[:target_samples]
        
        return noise
    
    def compute_snr_gain(self, speech: np.ndarray, noise: np.ndarray, target_snr_db: float) -> float:
        """
        Compute gain γ to achieve target SNR
        
        SNR (dB) = 10·log10(P_speech / P_noise)
        γ = sqrt(P_speech / (P_noise · 10^(SNR/10)))
        
        Args:
            speech: Clean speech signal
            noise: Noise signal
            target_snr_db: Target SNR in dB
        
        Returns:
            gamma: Gain factor for noise
        """
        # Compute power (RMS^2)
        p_speech = np.mean(speech**2)
        p_noise = np.mean(noise**2)
        
        # Avoid division by zero
        if p_noise < 1e-10:
            return 0.0
        
        # Compute gain from SNR
        snr_linear = 10**(target_snr_db / 10.0)
        gamma = np.sqrt(p_speech / (p_noise * snr_linear))
        
        return gamma
    
    def generate_mask(self, length: int, strategy: str = "stochastic") -> np.ndarray:
        """
        Generate time-varying mask m(t)
        
        Args:
            length: Number of samples
            strategy: 'stochastic', 'gaps', or 'ramp'
        
        Returns:
            mask: Array of [0, 1] values
        """
        if strategy == "stochastic":
            # Random time-varying mask
            # Generate smooth random values
            num_control_points = max(10, length // 1000)
            control_points = np.random.uniform(0.5, 1.0, num_control_points)
            # Interpolate to full length
            x = np.linspace(0, num_control_points - 1, length)
            mask = np.interp(x, np.arange(num_control_points), control_points)
            
        elif strategy == "gaps":
            # Random gaps where noise is reduced
            mask = np.ones(length)
            num_gaps = random.randint(3, 8)
            for _ in range(num_gaps):
                gap_start = random.randint(0, length - 1000)
                gap_len = random.randint(500, 2000)
                gap_end = min(gap_start + gap_len, length)
                mask[gap_start:gap_end] *= random.uniform(0.2, 0.5)
            
        elif strategy == "ramp":
            # Ramping mask (fade in/out)
            ramp_len = min(5000, length // 4)
            mask = np.ones(length)
            mask[:ramp_len] = np.linspace(0.3, 1.0, ramp_len)
            mask[-ramp_len:] = np.linspace(1.0, 0.3, ramp_len)
            
        else:
            # Default: constant mask
            mask = np.ones(length)
        
        return mask
    
    def mix(
        self,
        speech_path: str,
        scene: str,
        output_path: Optional[str] = None,
        snr_db: Optional[float] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Mix speech with scene-consistent noise
        
        Args:
            speech_path: Path to clean speech audio
            scene: Scene label for noise selection
            output_path: Optional path to save mixed audio
            snr_db: Optional fixed SNR (if None, randomly sample from range)
        
        Returns:
            mixed: Mixed audio array
            params: Dictionary of mixing parameters (for logging)
        """
        # Load speech
        speech, sr = self.load_audio(speech_path)
        duration = len(speech) / sr
        
        # Sample noise
        noise = self.sample_noise(scene, duration)
        
        # Determine SNR
        if snr_db is None:
            snr_db = random.uniform(self.snr_range[0], self.snr_range[1])
        
        # Compute gain
        gamma = self.compute_snr_gain(speech, noise, snr_db)
        
        # Generate mask
        mask = self.generate_mask(len(speech), self.masking_strategy)
        
        # Mix: x'(t) = x(t) + γ·n(t)·m(t)
        mixed = speech + gamma * noise * mask
        
        # Prevent clipping
        max_val = np.abs(mixed).max()
        if max_val > 0.99:
            mixed = mixed / max_val * 0.99
        
        # Save if output path provided
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            sf.write(output_path, mixed, sr)
        
        # Return mixing parameters for logging
        params = {
            'speech_path': speech_path,
            'scene': scene,
            'snr_db': float(snr_db),
            'gamma': float(gamma),
            'masking_strategy': self.masking_strategy,
            'output_path': output_path,
            'duration_sec': float(duration),
            'speech_rms': float(np.sqrt(np.mean(speech**2))),
            'noise_rms': float(np.sqrt(np.mean(noise**2))),
            'mixed_rms': float(np.sqrt(np.mean(mixed**2))),
        }
        
        return mixed, params
    
    def mix_optimized(
        self,
        speech_path: str,
        scene: str,
        speaker_encoder,
        output_path: Optional[str] = None,
        reference_text: Optional[str] = None,
        max_epochs: int = 50,
        lambda_sim: float = 1.0,
        lambda_reg: float = 0.01,
        lr: float = 0.01,
        log_path: Optional[str] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Mix speech with scene-consistent noise using gradient-based optimization
        
        This method jointly optimizes the temporal mask m(t) and noise strength γ
        to minimize speaker similarity while maintaining usability.
        
        Args:
            speech_path: Path to clean speech audio
            scene: Scene label for noise selection
            speaker_encoder: Pre-trained speaker verification model (ECAPA-TDNN)
            output_path: Optional path to save mixed audio
            reference_text: Optional reference text for ASR constraint
            max_epochs: Maximum optimization iterations
            lambda_sim: Weight for speaker similarity loss
            lambda_reg: Weight for regularization
            lr: Learning rate
            log_path: Optional path to save optimization log
        
        Returns:
            mixed: Optimized mixed audio array
            params: Dictionary of optimization results
        """
        from ..optimization import SceneGuardOptimizer
        
        # Load speech
        speech, sr = self.load_audio(speech_path)
        duration = len(speech) / sr
        
        # Sample noise
        noise = self.sample_noise(scene, duration)
        
        # Create optimizer
        optimizer = SceneGuardOptimizer(
            speaker_encoder=speaker_encoder,
            snr_range=self.snr_range,
            lambda_sim=lambda_sim,
            lambda_reg=lambda_reg,
            max_epochs=max_epochs,
            lr=lr,
            seed=self.seed
        )
        
        # Optimize
        print(f"\nOptimizing protection for: {speech_path}")
        print(f"  Scene: {scene}")
        print(f"  Duration: {duration:.2f}s")
        
        mixed, opt_result = optimizer.optimize(
            speech=speech,
            noise=noise,
            scene_label=scene,
            reference_text=reference_text,
            sr=sr,
            save_trajectory=True
        )
        
        # Save if output path provided
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            sf.write(output_path, mixed, sr)
        
        # Prepare params dictionary
        params = {
            'speech_path': speech_path,
            'scene': scene,
            'method': 'optimized',
            'final_snr_db': opt_result['final_snr'],
            'final_gamma': opt_result['final_gamma'],
            'final_similarity': opt_result['final_similarity'],
            'mask_mean': opt_result['mask_mean'],
            'mask_std': opt_result['mask_std'],
            'output_path': output_path,
            'duration_sec': float(duration),
            'speech_rms': float(opt_result['speech_rms']),
            'noise_rms': float(opt_result['noise_rms']),
            'mixed_rms': float(opt_result['protected_rms']),
            'optimization': {
                'max_epochs': max_epochs,
                'lambda_sim': lambda_sim,
                'lambda_reg': lambda_reg,
                'lr': lr,
            }
        }
        
        # Save optimization log if requested
        if log_path:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            
            log_dict = {
                **params,
                'trajectory': opt_result['trajectory']
            }
            
            with open(log_path, 'w') as f:
                json.dump(log_dict, f, indent=2)
        
        return mixed, params

