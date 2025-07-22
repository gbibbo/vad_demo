# src/models/epanns_model.py
import numpy as np
import torch
import os
import sys
from typing import Optional

# Añadir paths necesarios
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(project_root)

from src.core.base_vad import BaseVADModel

# Importar E-PANNs wrapper existente
try:
    from src.wrappers.vad_epanns import EPANNsVADWrapper, SPEECH_INDICES as EP_SPEECH_INDICES
    EPANNS_AVAILABLE = True
except ImportError as e:
    print(f"❌ E-PANNs wrapper not found: {e}")
    EPANNsVADWrapper = None
    EP_SPEECH_INDICES = None
    EPANNS_AVAILABLE = False

class EPANNsModel(BaseVADModel):
    """Implementación específica para E-PANNs usando el wrapper existente"""
    
    def __init__(self, 
                 checkpoint: str = "models/epanns/E-PANNs/models/checkpoint_closeto_.44.pt",
                 samplerate: int = 32000):
        
        super().__init__(name="E-PANNs", color="orange", samplerate=samplerate)
        
        # Convertir path relativo a absoluto
        if not os.path.isabs(checkpoint):
            self.checkpoint_path = os.path.join(project_root, checkpoint)
        else:
            self.checkpoint_path = checkpoint
        
        # Parámetros específicos de E-PANNs (similares a PANNs)
        self.model_winsize = 1024
        self.stft_hopsize = 512
        self.patch_frames = 32
        
        # Calcular muestras necesarias
        self.samples_needed = self.model_winsize + (self.patch_frames - 1) * self.stft_hopsize
        
        # Componentes del modelo
        self.epanns_wrapper: Optional[EPANNsVADWrapper] = None
    
    def initialize(self) -> bool:
        """Inicializa el modelo E-PANNs"""
        if not EPANNS_AVAILABLE:
            print("❌ E-PANNs wrapper not available")
            return False
        
        try:
            print(f"Loading E-PANNs model from {self.checkpoint_path}...")
            
            # Verificar que el checkpoint existe
            if not os.path.exists(self.checkpoint_path):
                print(f"❌ E-PANNs checkpoint not found: {self.checkpoint_path}")
                return False
            
            # Crear wrapper de E-PANNs usando el wrapper existente
            self.epanns_wrapper = EPANNsVADWrapper(checkpoint=self.checkpoint_path)
            
            print(f"✓ E-PANNs model loaded successfully")
            print(f"✓ E-PANNs ready - needs {self.samples_needed} samples per inference")
            self.is_initialized = True
            return True
            
        except Exception as e:
            print(f"❌ Error loading E-PANNs model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def process_audio(self, audio_chunk: np.ndarray) -> float:
        """Procesa audio y devuelve probabilidad de habla"""
        if not self.is_initialized or self.epanns_wrapper is None:
            return 0.0
        
        try:
            # Verificar que tenemos suficientes muestras
            if len(audio_chunk) < self.samples_needed:
                # Pad con ceros si es necesario
                padded_audio = np.concatenate([
                    np.zeros(self.samples_needed - len(audio_chunk)),
                    audio_chunk
                ])
            else:
                padded_audio = audio_chunk[-self.samples_needed:]
            
            # Ejecutar inferencia con E-PANNs usando el wrapper existente
            with torch.no_grad():
                predictions = self.epanns_wrapper.audio_inference(padded_audio)
                
                # Extraer probabilidades de habla usando los índices específicos
                if EP_SPEECH_INDICES is not None:
                    speech_probs = predictions[EP_SPEECH_INDICES]
                    max_prob = float(speech_probs.max())
                else:
                    # Fallback: usar toda la predicción
                    max_prob = float(predictions.max())
                
                return max_prob
            
        except Exception as e:
            print(f"[E-PANNs] Error processing audio: {e}")
            return 0.0
    
    def get_model_info(self) -> dict:
        """Devuelve información del modelo"""
        return {
            "name": self.name,
            "type": "Efficient CNN-based VAD",
            "checkpoint": self.checkpoint_path,
            "input_length": self.samples_needed,
            "window_size": self.model_winsize,
            "hop_size": self.stft_hopsize,
            "patch_frames": self.patch_frames,
            "samplerate": self.samplerate,
            "speech_indices": len(EP_SPEECH_INDICES) if EP_SPEECH_INDICES else "Unknown",
            "wrapper_available": EPANNS_AVAILABLE
        }
    
    def cleanup(self):
        """Limpia recursos específicos de E-PANNs"""
        super().cleanup()
        self.epanns_wrapper = None
        # Limpiar cache de GPU si está disponible
        if torch.cuda.is_available():
            torch.cuda.empty_cache()