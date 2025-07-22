# src/models/panns_model.py
import numpy as np
import torch
import os
import sys
from typing import Optional

# Añadir paths necesarios para acceder a la estructura existente
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(project_root)

from src.core.base_vad import BaseVADModel

# Importar módulos originales de PANNs desde sed_demo
from sed_demo.models import Cnn9_GMP_64x64
from sed_demo.utils import load_csv_labels
from sed_demo.inference import AudioModelInference

class PANNsModel(BaseVADModel):
    """Implementación específica para PANNs usando la estructura existente"""
    
    SPEECH_TAGS = {
        "Speech": 0,
        "Male speech, man speaking": 1,
        "Female speech, woman speaking": 2,
        "Child speech, kid speaking": 3,
        "Conversation": 4,
        "Narration, monologue": 5
    }
    
    def __init__(self, 
                 model_path: str = "Cnn9_GMP_64x64_300000_iterations_mAP=0.37.pth",
                 labels_path: str = "sed_demo/assets/audioset_labels.csv",
                 samplerate: int = 32000):
        
        super().__init__(name="PANNs", color="cyan", samplerate=samplerate)
        
        # Convertir paths relativos a absolutos
        if not os.path.isabs(model_path):
            self.model_path = os.path.join(project_root, model_path)
        else:
            self.model_path = model_path
            
        if not os.path.isabs(labels_path):
            self.labels_path = os.path.join(project_root, labels_path)
        else:
            self.labels_path = labels_path
        
        # Parámetros específicos de PANNs (coinciden con el código original)
        self.model_winsize = 1024
        self.stft_hopsize = 512
        self.n_mels = 64
        self.patch_frames = 32
        
        # Calcular muestras necesarias (igual que en el código original)
        self.samples_needed = self.model_winsize + (self.patch_frames - 1) * self.stft_hopsize
        
        # Componentes del modelo
        self.model: Optional[Cnn9_GMP_64x64] = None
        self.inference: Optional[AudioModelInference] = None
        self.all_labels: Optional[list] = None
        self.speech_label_indices: Optional[list] = None
    
    def initialize(self) -> bool:
        """Inicializa el modelo PANNs"""
        try:
            print(f"Loading PANNs model from {self.model_path}...")
            
            # Verificar que los archivos existen
            if not os.path.exists(self.model_path):
                print(f"❌ PANNs checkpoint not found: {self.model_path}")
                return False
                
            if not os.path.exists(self.labels_path):
                print(f"❌ Labels file not found: {self.labels_path}")
                return False
            
            # Cargar etiquetas usando la función del sed_demo original
            _, _, self.all_labels = load_csv_labels(self.labels_path)
            self.speech_label_indices = [
                self.all_labels.index(label) 
                for label in self.SPEECH_TAGS 
                if label in self.all_labels
            ]
            
            if not self.speech_label_indices:
                print(f"❌ No speech labels found in {self.labels_path}")
                return False
            
            # Crear y cargar modelo usando la clase del sed_demo original
            self.model = Cnn9_GMP_64x64(len(self.all_labels))
            
            try:
                checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=False)
                self.model.load_state_dict(checkpoint["model"])
                print(f"✓ PANNs model loaded successfully")
            except Exception as e:
                print(f"❌ Error loading checkpoint: {e}")
                return False
            
            # Crear objeto de inferencia usando la clase del sed_demo original
            self.inference = AudioModelInference(
                self.model,
                winsize=self.model_winsize,
                stft_hopsize=self.stft_hopsize,
                samplerate=self.samplerate,
                stft_window="hann"
            )
            
            self.is_initialized = True
            print(f"✓ PANNs ready - needs {self.samples_needed} samples per inference")
            return True
            
        except Exception as e:
            print(f"❌ Error loading PANNs model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def process_audio(self, audio_chunk: np.ndarray) -> float:
        """Procesa audio y devuelve probabilidad de habla"""
        if not self.is_initialized or self.inference is None:
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
            
            # Ejecutar inferencia usando el objeto original de sed_demo
            predictions = self.inference(padded_audio)
            
            # Extraer probabilidades de habla
            speech_vec = predictions[self.speech_label_indices]
            max_prob = float(speech_vec.max())
            
            return max_prob
            
        except Exception as e:
            print(f"[PANNs] Error processing audio: {e}")
            return 0.0
    
    def get_model_info(self) -> dict:
        """Devuelve información del modelo"""
        return {
            "name": self.name,
            "type": "CNN-based Audio Tagging",
            "checkpoint": self.model_path,
            "labels_file": self.labels_path,
            "input_length": self.samples_needed,
            "window_size": self.model_winsize,
            "hop_size": self.stft_hopsize,
            "mel_bins": self.n_mels,
            "patch_frames": self.patch_frames,
            "speech_classes": len(self.speech_label_indices) if self.speech_label_indices else 0,
            "total_classes": len(self.all_labels) if self.all_labels else 0,
            "samplerate": self.samplerate
        }
    
    def cleanup(self):
        """Limpia recursos específicos de PANNs"""
        super().cleanup()
        self.model = None
        self.inference = None
        # Limpiar cache de GPU si está disponible
        if torch.cuda.is_available():
            torch.cuda.empty_cache()