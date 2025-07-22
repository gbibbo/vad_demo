# src/core/audio_processor.py
import numpy as np
import librosa
import queue
import threading
import time
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
import pyaudio
import sys
import os

# Importar desde tu estructura existente
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(project_root)
from sed_demo.audio_loop import AsynchAudioInputStream

@dataclass
class AudioConfig:
    """Configuración de audio"""
    samplerate: int = 32000
    chunk_length: int = 1024
    ringbuffer_length: int = 38400  # 1.2 segundos a 32kHz
    
    # Parámetros para visualización
    display_n_fft: int = 2048
    display_hop_length: int = 512
    display_n_mels: int = 128
    display_fmin: float = 20
    display_fmax: float = 8000
    
    # Parámetros de visualización temporal
    display_duration: float = 10.0
    fps: int = 10
    
    @property
    def spec_time_resolution(self) -> float:
        return self.display_hop_length / self.samplerate
    
    @property
    def spec_buffer_size(self) -> int:
        return int(self.display_duration / self.spec_time_resolution)

@dataclass
class AudioFrame:
    """Frame de audio procesado"""
    timestamp: float
    mel_spectrogram: np.ndarray
    frame_db: np.ndarray
    raw_audio: np.ndarray

class MyAudioInputStream(AsynchAudioInputStream):
    """Extended audio stream with sample counting"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.total_samples_written = 0

    def callback(self, in_data, frame_count, time_info, status):
        super().callback(in_data, frame_count, time_info, status)
        self.total_samples_written += frame_count
        return (in_data, pyaudio.paContinue)
    
    def get_total_samples_written(self):
        return self.total_samples_written

class AudioProcessor:
    """Maneja el procesamiento de audio en tiempo real"""
    
    def __init__(self, config: Optional[AudioConfig] = None):
        self.config = config or AudioConfig()
        
        # Estado del procesador
        self.is_running = False
        self.time_cursor = 0.0
        self._residual_audio = np.array([], dtype=np.float32)
        self.last_total_samples = 0
        
        # Queues para comunicación entre threads
        self.audio_queue = queue.Queue(maxsize=100)
        self.frame_queue = queue.Queue(maxsize=50)
        
        # Threading
        self.processing_thread: Optional[threading.Thread] = None
        
        # Callbacks
        self.frame_callbacks: list[Callable[[AudioFrame], None]] = []
        
        # Audio stream
        self.audiostream: Optional[MyAudioInputStream] = None
        
        # Debugging - detección de energía
        self.energy_db_thresh = -35.0
        self.energy_prev_high = False
        self.energy_on_time: Optional[float] = None
        
    def add_frame_callback(self, callback: Callable[[AudioFrame], None]):
        """Añade callback que se ejecuta por cada frame procesado"""
        self.frame_callbacks.append(callback)
    
    def remove_frame_callback(self, callback: Callable[[AudioFrame], None]):
        """Remueve callback"""
        if callback in self.frame_callbacks:
            self.frame_callbacks.remove(callback)
    
    def initialize(self) -> bool:
        """Inicializa el stream de audio"""
        try:
            print("Initializing audio stream...")
            self.audiostream = MyAudioInputStream(
                samplerate=self.config.samplerate,
                chunk_length=self.config.chunk_length,
                ringbuffer_length=self.config.ringbuffer_length
            )
            print("✓ Audio stream ready")
            return True
        except Exception as e:
            print(f"❌ Error initializing audio stream: {e}")
            return False
    
    def start(self) -> bool:
        """Inicia el procesamiento de audio"""
        if self.is_running:
            return True
            
        if not self.audiostream:
            if not self.initialize():
                return False
        
        print("\n🎤 Starting audio processing...")
        
        # Reset state
        self.is_running = True
        self.time_cursor = 0.0
        self._residual_audio = np.array([], dtype=np.float32)
        self.last_total_samples = 0
        self.energy_prev_high = False
        self.energy_on_time = None
        
        # Clear queues
        while not self.audio_queue.empty():
            self.audio_queue.get()
        while not self.frame_queue.empty():
            self.frame_queue.get()
        
        # Start audio stream
        self.audiostream.start()
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        print("✓ Audio processing started")
        return True
    
    def stop(self):
        """Detiene el procesamiento de audio"""
        if not self.is_running:
            return
            
        print("\n⏹️ Stopping audio processing...")
        self.is_running = False
        
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
        
        if self.audiostream:
            self.audiostream.stop()
        
        print("✓ Audio processing stopped")
    
    def cleanup(self):
        """Limpia recursos"""
        self.stop()
        if self.audiostream:
            self.audiostream.terminate()
            self.audiostream = None
    
    def get_audio_for_model(self, samples_needed: int) -> Optional[np.ndarray]:
        """Obtiene audio para un modelo específico"""
        if not self.audiostream:
            return None
            
        full_ring_buffer = self.audiostream.read()
        
        if len(full_ring_buffer) < samples_needed:
            # Pad with zeros if needed
            recent_audio = np.concatenate((
                np.zeros(samples_needed - len(full_ring_buffer)),
                full_ring_buffer
            ))
        else:
            recent_audio = full_ring_buffer[-samples_needed:]
            
        return recent_audio
    
    def get_display_data(self) -> Optional[Dict[str, Any]]:
        """Obtiene datos para visualización"""
        try:
            return self.audio_queue.get_nowait()
        except queue.Empty:
            return None
    
    def get_current_time(self) -> float:
        """Obtiene el timestamp actual"""
        return self.time_cursor
    
    def _processing_loop(self):
        """Loop principal de procesamiento de audio"""
        print("✓ Audio processing loop started")
        
        while self.is_running:
            try:
                if not self.audiostream:
                    time.sleep(0.02)
                    continue
                    
                full_ring_buffer = self.audiostream.read()
                n_frames = 0
                
                # Procesar nuevas muestras para espectrograma
                new_sample_count = (self.audiostream.get_total_samples_written() - 
                                  self.last_total_samples)
                
                if new_sample_count > 0:
                    self._residual_audio = np.concatenate((
                        self._residual_audio,
                        full_ring_buffer[-new_sample_count:]
                    ))
                    self.last_total_samples += new_sample_count
                    
                    # Procesar espectrograma si hay suficientes muestras
                    if len(self._residual_audio) >= self.config.display_n_fft:
                        n_frames = self._process_spectrogram()
                
                # Llamar callbacks con cada frame
                if n_frames > 0:
                    self._call_frame_callbacks(n_frames)
                
                time.sleep(0.02)  # ~50 FPS para el loop de audio
                
            except Exception as e:
                print(f"[AudioProcessor] Warning → {e}")
                continue
    
    def _process_spectrogram(self) -> int:
        """Procesa el espectrograma y actualiza las queues"""
        mel_spec = librosa.feature.melspectrogram(
            y=self._residual_audio,
            sr=self.config.samplerate,
            n_fft=self.config.display_n_fft,
            hop_length=self.config.display_hop_length,
            n_mels=self.config.display_n_mels,
            fmin=self.config.display_fmin,
            fmax=self.config.display_fmax
        )
        
        n_frames = mel_spec.shape[1]
        samples_processed = n_frames * self.config.display_hop_length
        
        if n_frames > 0:
            # Datos para visualización
            spec_db = np.clip(
                librosa.power_to_db(mel_spec, ref=1.0),
                -60, -10  # spec_vmin, spec_vmax
            )
            
            self.audio_queue.put({
                'spec_display': spec_db,
                'timestamp': self.time_cursor
            })
            
            # Calcular energía por frame
            frame_db = librosa.power_to_db(mel_spec.mean(axis=0), ref=1.0)
            
            # Actualizar cursor temporal
            self.time_cursor += n_frames * self.config.spec_time_resolution
            
            # Crear frames individuales
            for i in range(n_frames):
                frame_time = self.time_cursor - (n_frames - i) * self.config.spec_time_resolution
                
                frame = AudioFrame(
                    timestamp=frame_time,
                    mel_spectrogram=mel_spec[:, i:i+1],
                    frame_db=np.array([frame_db[i]]),
                    raw_audio=self._residual_audio[i*self.config.display_hop_length:
                                                 (i+1)*self.config.display_hop_length]
                )
                
                try:
                    self.frame_queue.put_nowait(frame)
                except queue.Full:
                    # Skip frame if queue is full
                    pass
            
            # Remover muestras procesadas
            self._residual_audio = self._residual_audio[samples_processed:]
            
        return n_frames
    
    def _call_frame_callbacks(self, n_frames: int):
        """Llama a todos los callbacks registrados"""
        # Procesar frames de la queue
        frames_processed = 0
        while frames_processed < n_frames and not self.frame_queue.empty():
            try:
                frame = self.frame_queue.get_nowait()
                
                # Detectar onset de energía para calibración
                self._detect_energy_onset(frame)
                
                # Llamar a todos los callbacks
                for callback in self.frame_callbacks:
                    try:
                        callback(frame)
                    except Exception as e:
                        print(f"[AudioProcessor] Callback error: {e}")
                
                frames_processed += 1
                
            except queue.Empty:
                break
    
    def _detect_energy_onset(self, frame: AudioFrame):
        """Detecta onset de energía para calibración de delays"""
        energy_db = frame.frame_db[0]
        energy_high = energy_db > self.energy_db_thresh
        
        if energy_high and not self.energy_prev_high:
            print(f"[DEBUG] 📈 Energy ON @ {frame.timestamp:6.3f}s (dB={energy_db:.1f})")
            self.energy_on_time = frame.timestamp
            
        self.energy_prev_high = energy_high
    
    def get_energy_onset_time(self) -> Optional[float]:
        """Obtiene el tiempo del último onset de energía"""
        return self.energy_on_time