"""
Emotion Prediction Test Suite

This package contains utilities for testing emotion recognition models
on audio files.
"""

__version__ = "1.0.0"

from .vit_model import ViTForEmotionRegression, MobileViTStudent
from .audio_preprocessor import AudioPreprocessor
from .predict import EmotionPredictor

__all__ = [
    'ViTForEmotionRegression',
    'MobileViTStudent', 
    'AudioPreprocessor',
    'EmotionPredictor'
]
