"""
SceneGuard Evaluation Module
Metrics calculation for defense effectiveness
"""

from .metrics import compute_speaker_similarity, compute_wer

__all__ = ['compute_speaker_similarity', 'compute_wer']

