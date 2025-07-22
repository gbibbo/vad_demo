# src/core/base_vad.py
from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, List, Dict, Any
from collections import deque
from dataclasses import dataclass

@dataclass
class VADEvent:
    """Representa un evento de VAD (onset/offset)"""
    event_type: str  # 'on' o 'off'
    timestamp: float
    probability: float

@dataclass
class VADState:
    """Estado actual del VAD"""
    is_active: bool = False
    current_probability: float = 0.0
    events: List[VADEvent] = None
    
    def __post_init__(self):
        if self.events is None:
            self.events = []

class BaseVADModel(ABC):
    """Clase base abstracta para todos los modelos VAD"""
    
    def __init__(self, name: str, color: str, samplerate: int = 32000):
        self.name = name
        self.color = color  # Color para visualización
        self.samplerate = samplerate
        self.threshold = 0.3
        
        # Estado interno
        self.state = VADState()
        self.prob_history = deque(maxlen=1000)  # Ajustable según buffer_size
        self.delay = 0.0  # Delay calibrado en segundos
        
        # Configuración del modelo
        self.is_initialized = False
        self.samples_needed = 0  # Número de muestras que necesita el modelo
        
    @abstractmethod
    def initialize(self) -> bool:
        """Inicializa el modelo. Debe ser implementado por cada modelo específico."""
        pass
    
    @abstractmethod
    def process_audio(self, audio_chunk: np.ndarray) -> float:
        """Procesa un chunk de audio y devuelve la probabilidad de habla.
        
        Args:
            audio_chunk: Array de audio
            
        Returns:
            float: Probabilidad de habla [0, 1]
        """
        pass
    
    def update_state(self, probability: float, timestamp: float) -> None:
        """Actualiza el estado del VAD basado en la probabilidad"""
        self.prob_history.append(probability)
        self.state.current_probability = probability
        
        # Detectar eventos (onset/offset)
        was_active = self.state.is_active
        self.state.is_active = probability >= self.threshold
        
        if not was_active and self.state.is_active:
            # Onset detectado
            event = VADEvent('on', timestamp, probability)
            self.state.events.append(event)
            self._log_event(event)
            
        elif was_active and not self.state.is_active:
            # Offset detectado
            event = VADEvent('off', timestamp, probability)
            self.state.events.append(event)
            self._log_event(event)
    
    def _log_event(self, event: VADEvent):
        """Log de eventos para debugging"""
        icon = "🎤" if event.event_type == 'on' else "🔇"
        print(f"{icon} {self.name} {event.event_type.upper()} @ {event.timestamp:.3f}s p:{event.probability*100:.1f}%")
    
    def set_threshold(self, threshold: float) -> None:
        """Establece el umbral de detección"""
        self.threshold = np.clip(threshold, 0.0, 1.0)
    
    def get_current_probability(self) -> float:
        """Obtiene la probabilidad actual"""
        return self.state.current_probability
    
    def get_probability_history(self) -> List[float]:
        """Obtiene el historial de probabilidades"""
        return list(self.prob_history)
    
    def get_recent_events(self, window_start: float, window_end: float) -> List[VADEvent]:
        """Obtiene eventos dentro de una ventana temporal"""
        return [event for event in self.state.events 
                if window_start <= event.timestamp <= window_end]
    
    def set_delay(self, delay: float) -> None:
        """Establece el delay calibrado para este modelo"""
        self.delay = delay
        print(f"[{self.name}] Delay set to {delay*1000:.0f} ms")
    
    def get_corrected_timestamp(self, timestamp: float, apply_correction: bool = True) -> float:
        """Aplica corrección de delay al timestamp"""
        if apply_correction:
            return timestamp - self.delay
        return timestamp
    
    def cleanup(self) -> None:
        """Limpia recursos del modelo"""
        self.is_initialized = False
        self.state = VADState()
        self.prob_history.clear()
    
    def reset_state(self) -> None:
        """Reinicia el estado para una nueva sesión"""
        self.state = VADState()
        self.prob_history.clear()
        self.delay = 0.0
    
    def get_status_text(self) -> str:
        """Devuelve texto de estado para la GUI"""
        status = "ACTIVE" if self.state.is_active else "INACTIVE"
        return f"{self.name}: {status}"
    
    def get_status_color(self) -> str:
        """Devuelve color de estado para la GUI"""
        return 'lime' if self.state.is_active else 'red'
    
    def __str__(self) -> str:
        return f"{self.name}VAD(active={self.state.is_active}, prob={self.state.current_probability:.3f})"