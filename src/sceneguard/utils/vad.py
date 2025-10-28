"""
Voice Activity Detection (VAD) Module

Detects speech/silence segments for identifying candidate noise insertion positions.
Uses energy-based VAD as a lightweight alternative to deep learning models.
"""

import numpy as np
import librosa
from typing import List, Tuple, Optional


class VoiceActivityDetector:
    """
    Energy-based Voice Activity Detector
    
    Identifies speech and silence segments based on short-term energy and
    zero-crossing rate.
    
    Args:
        frame_length: Frame length in samples (default: 400 samples ~= 25ms @ 16kHz)
        hop_length: Hop length in samples (default: 160 samples ~= 10ms @ 16kHz)
        energy_threshold: Relative energy threshold (0-1, default: 0.02)
        min_silence_duration: Minimum silence duration in seconds (default: 0.3s)
    """
    
    def __init__(
        self,
        frame_length: int = 400,
        hop_length: int = 160,
        energy_threshold: float = 0.02,
        min_silence_duration: float = 0.3
    ):
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.energy_threshold = energy_threshold
        self.min_silence_duration = min_silence_duration
    
    def compute_frame_energy(self, audio: np.ndarray) -> np.ndarray:
        """
        Compute short-term energy for each frame
        
        Args:
            audio: Audio signal [T]
        
        Returns:
            energy: Frame-level energy [N_frames]
        """
        # Pad audio
        padded = np.pad(audio, (self.frame_length // 2, self.frame_length // 2), mode='reflect')
        
        # Compute energy for each frame
        num_frames = (len(padded) - self.frame_length) // self.hop_length + 1
        energy = np.zeros(num_frames)
        
        for i in range(num_frames):
            start = i * self.hop_length
            end = start + self.frame_length
            frame = padded[start:end]
            energy[i] = np.sum(frame ** 2)
        
        return energy
    
    def detect_segments(
        self,
        audio: np.ndarray,
        sr: int = 16000
    ) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """
        Detect speech and silence segments
        
        Args:
            audio: Audio signal [T]
            sr: Sampling rate
        
        Returns:
            speech_segments: List of (start_time, end_time) for speech
            silence_segments: List of (start_time, end_time) for silence
        """
        # Compute frame-level energy
        energy = self.compute_frame_energy(audio)
        
        # Normalize energy
        max_energy = np.max(energy)
        if max_energy > 0:
            energy_norm = energy / max_energy
        else:
            energy_norm = energy
        
        # Detect speech frames (above threshold)
        speech_frames = energy_norm > self.energy_threshold
        
        # Convert frame indices to time
        frame_to_time = lambda frame_idx: frame_idx * self.hop_length / sr
        
        # Extract segments
        speech_segments = []
        silence_segments = []
        
        in_speech = False
        segment_start = 0
        
        for i in range(len(speech_frames)):
            if speech_frames[i] and not in_speech:
                # Start of speech
                if i > 0:
                    # End previous silence
                    silence_end = frame_to_time(i)
                    if silence_end - frame_to_time(segment_start) >= self.min_silence_duration:
                        silence_segments.append((frame_to_time(segment_start), silence_end))
                
                segment_start = i
                in_speech = True
            
            elif not speech_frames[i] and in_speech:
                # End of speech, start of silence
                speech_end = frame_to_time(i)
                speech_segments.append((frame_to_time(segment_start), speech_end))
                
                segment_start = i
                in_speech = False
        
        # Handle final segment
        if in_speech:
            speech_segments.append((frame_to_time(segment_start), frame_to_time(len(speech_frames))))
        else:
            silence_end = frame_to_time(len(speech_frames))
            if silence_end - frame_to_time(segment_start) >= self.min_silence_duration:
                silence_segments.append((frame_to_time(segment_start), silence_end))
        
        return speech_segments, silence_segments
    
    def get_candidate_noise_positions(
        self,
        audio: np.ndarray,
        sr: int = 16000,
        max_candidates: int = 5
    ) -> List[int]:
        """
        Get candidate positions (in samples) for noise insertion
        
        Prioritizes silence segments as they have lower risk of degrading WER.
        
        Args:
            audio: Audio signal [T]
            sr: Sampling rate
            max_candidates: Maximum number of candidate positions
        
        Returns:
            positions: List of sample indices for noise insertion
        """
        _, silence_segments = self.detect_segments(audio, sr)
        
        if len(silence_segments) == 0:
            # No clear silence detected, return uniformly spaced positions
            duration = len(audio) / sr
            positions = [int(i * len(audio) / (max_candidates + 1)) 
                        for i in range(1, max_candidates + 1)]
            return positions
        
        # Extract middle of each silence segment
        positions = []
        for start_time, end_time in silence_segments[:max_candidates]:
            mid_time = (start_time + end_time) / 2.0
            mid_sample = int(mid_time * sr)
            positions.append(mid_sample)
        
        # If fewer than max_candidates, add uniform positions
        while len(positions) < max_candidates:
            # Add position between existing ones
            if len(positions) == 0:
                positions.append(len(audio) // 2)
            else:
                # Find largest gap
                sorted_pos = sorted(positions + [0, len(audio)])
                max_gap = 0
                max_gap_pos = 0
                for i in range(len(sorted_pos) - 1):
                    gap = sorted_pos[i+1] - sorted_pos[i]
                    if gap > max_gap:
                        max_gap = gap
                        max_gap_pos = (sorted_pos[i] + sorted_pos[i+1]) // 2
                positions.append(max_gap_pos)
        
        return sorted(positions[:max_candidates])


def detect_speech_segments(
    audio: np.ndarray,
    sr: int = 16000,
    energy_threshold: float = 0.02,
    min_silence_duration: float = 0.3
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """
    Convenience function: detect speech and silence segments
    
    Args:
        audio: Audio signal [T]
        sr: Sampling rate
        energy_threshold: Relative energy threshold
        min_silence_duration: Minimum silence duration in seconds
    
    Returns:
        speech_segments: List of (start_time, end_time) for speech
        silence_segments: List of (start_time, end_time) for silence
    """
    vad = VoiceActivityDetector(
        energy_threshold=energy_threshold,
        min_silence_duration=min_silence_duration
    )
    return vad.detect_segments(audio, sr)

