# src/core/delay_calibrator.py
import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict
from dataclasses import dataclass

@dataclass
class DelayMeasurement:
    """Medición individual de delay"""
    model_name: str
    energy_time: float
    vad_time: float
    lag: float
    
    def __post_init__(self):
        self.lag = self.vad_time - self.energy_time

class DelayCalibrator:
    """Maneja la calibración de delays para múltiples modelos VAD"""
    
    def __init__(self, prob_threshold: float = 0.50, valid_lag_max: float = 1.0):
        self.prob_threshold = prob_threshold
        self.valid_lag_max = valid_lag_max  # Máximo lag válido en segundos
        
        # Estado de tracking
        self.prob_prev_high: Dict[str, bool] = {}
        self.energy_onset_time: Optional[float] = None
        
        # Almacenamiento de mediciones
        self.measurements: Dict[str, List[float]] = defaultdict(list)
        self.calibrated_delays: Dict[str, float] = {}
        
        # Estado de calibración
        self.is_enabled = False
        
    def reset(self):
        """Reinicia el calibrador para una nueva sesión"""
        self.prob_prev_high.clear()
        self.measurements.clear()
        self.calibrated_delays.clear()
        self.energy_onset_time = None
        print("[Calibrator] State reset")
    
    def set_energy_onset(self, timestamp: float):
        """Establece el tiempo de onset de energía"""
        self.energy_onset_time = timestamp
    
    def update_model_probability(self, model_name: str, probability: float, timestamp: float):
        """Actualiza la probabilidad de un modelo y detecta onsets"""
        # Inicializar estado si es necesario
        if model_name not in self.prob_prev_high:
            self.prob_prev_high[model_name] = False
        
        # Detectar transición de low -> high
        prob_high = probability >= self.prob_threshold
        was_high = self.prob_prev_high[model_name]
        
        if prob_high and not was_high:
            # Onset detectado
            self._measure_lag(model_name, timestamp)
        
        self.prob_prev_high[model_name] = prob_high
    
    def _measure_lag(self, model_name: str, vad_time: float):
        """Mide el lag entre energy onset y VAD onset"""
        if self.energy_onset_time is None:
            return
        
        lag = vad_time - self.energy_onset_time
        
        # Filtrar lags válidos
        if 0 <= lag <= self.valid_lag_max:
            self.measurements[model_name].append(lag)
            print(f"[Calibrator] 🔔 {model_name} VAD ON @ {vad_time:6.3f}s. Δ={lag*1000:.0f} ms")
        else:
            print(f"[Calibrator] ⚠️ {model_name} Invalid lag: {lag*1000:.0f} ms (filtered)")
    
    def calibrate_model(self, model_name: str) -> Optional[float]:
        """Calibra un modelo específico basado en las mediciones"""
        if model_name not in self.measurements or not self.measurements[model_name]:
            print(f"[Calibrator] No measurements for {model_name}")
            return None
        
        measurements = self.measurements[model_name]
        
        # Calcular delay promedio (negativo para compensar)
        avg_lag = np.mean(measurements)
        delay = -avg_lag  # Negativo para adelantar el modelo
        
        self.calibrated_delays[model_name] = delay
        
        print(f"[Calibrator] {model_name} calibrated: "
              f"{len(measurements)} measurements, "
              f"avg lag: {avg_lag*1000:.0f} ms, "
              f"correction: {-delay*1000:.0f} ms")
        
        return delay
    
    def calibrate_all_models(self, model_names: List[str]) -> Dict[str, float]:
        """Calibra todos los modelos que tienen mediciones"""
        results = {}
        
        for model_name in model_names:
            delay = self.calibrate_model(model_name)
            if delay is not None:
                results[model_name] = delay
        
        return results
    
    def get_model_delay(self, model_name: str) -> float:
        """Obtiene el delay calibrado para un modelo"""
        return self.calibrated_delays.get(model_name, 0.0)
    
    def apply_delay_correction(self, model_name: str, timestamp: float) -> float:
        """Aplica corrección de delay a un timestamp"""
        if not self.is_enabled:
            return timestamp
        
        delay = self.get_model_delay(model_name)
        return timestamp - delay
    
    def enable_correction(self, enabled: bool = True):
        """Habilita/deshabilita la corrección de delay"""
        self.is_enabled = enabled
        
        if enabled:
            print("[Calibrator] Delay correction ENABLED")
            # Mostrar delays actuales
            for model_name, delay in self.calibrated_delays.items():
                print(f"  {model_name}: {-delay*1000:.0f} ms correction")
        else:
            print("[Calibrator] Delay correction DISABLED")
    
    def get_statistics(self) -> Dict[str, Dict]:
        """Obtiene estadísticas de calibración"""
        stats = {}
        
        for model_name, measurements in self.measurements.items():
            if measurements:
                stats[model_name] = {
                    'count': len(measurements),
                    'mean_lag_ms': np.mean(measurements) * 1000,
                    'std_lag_ms': np.std(measurements) * 1000,
                    'min_lag_ms': np.min(measurements) * 1000,
                    'max_lag_ms': np.max(measurements) * 1000,
                    'calibrated_delay_ms': -self.calibrated_delays.get(model_name, 0.0) * 1000
                }
        
        return stats
    
    def has_measurements(self, model_name: str) -> bool:
        """Verifica si hay mediciones para un modelo"""
        return model_name in self.measurements and len(self.measurements[model_name]) > 0
    
    def get_measurement_count(self, model_name: str) -> int:
        """Obtiene el número de mediciones para un modelo"""
        return len(self.measurements.get(model_name, []))
    
    def clear_model_measurements(self, model_name: str):
        """Limpia las mediciones de un modelo específico"""
        if model_name in self.measurements:
            self.measurements[model_name].clear()
        if model_name in self.calibrated_delays:
            del self.calibrated_delays[model_name]
        print(f"[Calibrator] Cleared measurements for {model_name}")