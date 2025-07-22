# src/__init__.py
"""
Módulo principal del sistema VAD modular
"""

__version__ = "1.0.0"
__author__ = "VAD Team"

# Importaciones principales
from .core.model_factory import model_factory
from .core.audio_processor import AudioProcessor, AudioConfig
from .core.delay_calibrator import DelayCalibrator

__all__ = [
    "model_factory",
    "AudioProcessor", 
    "AudioConfig",
    "DelayCalibrator"
]
