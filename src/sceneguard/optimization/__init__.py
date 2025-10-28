"""
SceneGuard Optimization Module

Implements gradient-based optimization for scene-consistent noise mixing.
Jointly optimizes temporal mask m(t) and noise strength γ to minimize
speaker similarity while maintaining usability.
"""

from .optimizer import SceneGuardOptimizer
from .losses import (
    compute_speaker_similarity_loss,
    compute_asr_loss,
    compute_scene_consistency_loss,
    compute_regularization_loss,
)

__all__ = [
    'SceneGuardOptimizer',
    'compute_speaker_similarity_loss',
    'compute_asr_loss',
    'compute_scene_consistency_loss',
    'compute_regularization_loss',
]

