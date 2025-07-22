# config/settings.py
"""
Configuraciones globales del sistema VAD modular
"""

import os
from pathlib import Path

# Paths del proyecto
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
SED_DEMO_DIR = PROJECT_ROOT / "sed_demo"

# Configuración de audio por defecto
AUDIO_CONFIG = {
    "samplerate": 32000,
    "chunk_length": 1024,
    "display_duration": 10.0,
    "fps": 10
}

# Paths de modelos específicos
MODEL_PATHS = {
    "panns": {
        "checkpoint": PROJECT_ROOT / "Cnn9_GMP_64x64_300000_iterations_mAP=0.37.pth",
        "labels": SED_DEMO_DIR / "assets" / "audioset_labels.csv"
    },
    "epanns": {
        "checkpoint": MODELS_DIR / "epanns" / "E-PANNs" / "models" / "checkpoint_closeto_.44.pt"
    }
}

# Configuración de visualización
VISUALIZATION_CONFIG = {
    "spec_vmin": -60,
    "spec_vmax": -10,
    "display_n_mels": 128,
    "display_fmin": 20,
    "display_fmax": 8000
}
