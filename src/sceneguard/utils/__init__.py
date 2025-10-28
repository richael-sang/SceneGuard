"""
SceneGuard Utilities

Helper functions and modules.
"""

from .vad import detect_speech_segments, VoiceActivityDetector

__all__ = [
    'detect_speech_segments',
    'VoiceActivityDetector',
]

