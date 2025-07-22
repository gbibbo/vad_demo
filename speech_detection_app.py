#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# speech_detection_panns_epanns_con_delay.py

"""
Real-time Speech Detection with Live Spectrogram Visualization
(Final Version) Implements model-specific, dynamically calibrated delay correction.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib.animation import FuncAnimation
from collections import deque
import threading
import time
import queue
import pyaudio
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
import wave, json, datetime

# Check for librosa
try:
    import librosa
    import librosa.display
except ImportError:
    print("❌ Error: librosa is required but not installed.")
    print("Please install it with: pip install librosa")
    sys.exit(1)

### FIX: Importar webrtcvad
try:
    import webrtcvad
except ImportError:
    print("❌ Error: webrtcvad-wheels is required but not installed.")
    print("Please install it with: pip install webrtcvad-wheels")
    sys.exit(1)

# Import original modules from sed_demo
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
from sed_demo.models import Cnn9_GMP_64x64
from sed_demo.utils import load_csv_labels
from sed_demo.inference import AudioModelInference
from sed_demo.audio_loop import AsynchAudioInputStream

# --- FIX: Add path to EPANNs module ---
root_dir = os.path.dirname(__file__)
wrappers_dir = os.path.join(root_dir, 'src', 'wrappers')
sys.path.append(wrappers_dir)

from vad_epanns import EPANNsVADWrapper, SPEECH_INDICES as EP_SPEECH_INDICES

import torch
from transformers import AutoProcessor, ASTForAudioClassification
from silero_vad import load_silero_vad

torch.set_num_threads(4) 

# Global constants for delay correction
APPLY_DELAY_CORRECTION = False
MAX_LAGS = 30
VALID_LAG_MAX = 1.0

class MyAudioInputStream(AsynchAudioInputStream):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.total_samples_written = 0

    def callback(self, in_data, frame_count, time_info, status):
        super().callback(in_data, frame_count, time_info, status)
        self.total_samples_written += frame_count
        return (in_data, pyaudio.paContinue)

    def get_total_samples_written(self):
        return self.total_samples_written

class SpeechDetectionApp:
    SPEECH_TAGS = {"Speech": 0, "Male speech, man speaking": 1, "Female speech, woman speaking": 2, "Child speech, kid speaking": 3, "Conversation": 4, "Narration, monologue": 5}
    
    def __init__(self, model_path="Cnn9_GMP_64x64_300000_iterations_mAP=0.37.pth", labels_path="sed_demo/assets/audioset_labels.csv"):
        # Parameters
        self.samplerate = 32000; self.audio_chunk_length = 1024
        self.model_winsize, self.stft_hopsize, self.n_mels = 1024, 512, 64
        self.patch_frames = 32; self.samples_needed = self.model_winsize + (self.patch_frames - 1) * self.stft_hopsize
        self.ringbuffer_length = max(int(self.samplerate * 1.2), self.samples_needed)
        
        self.display_n_fft, self.display_hop_length, self.display_n_mels, self.display_fmin, self.display_fmax = 2048, 512, 128, 20, 8000
        self.spec_vmin, self.spec_vmax = -60, -10; self.display_duration, self.fps = 10.0, 20
        self.spec_time_resolution = self.display_hop_length / self.samplerate
        self.spec_buffer_size = int(self.display_duration / self.spec_time_resolution)
        
        # Buffer para guardar audio
        self.snap_buffer = deque(maxlen=self.samplerate * 10)
        
        self.last_event_time = {'on': -np.inf, 'off': -np.inf}; self.min_event_gap   = 0.08

        # Data Buffers & State
        self.spec_data = np.full((self.display_n_mels, self.spec_buffer_size), self.spec_vmin)
        self.prob_history = deque(maxlen=self.spec_buffer_size); self.is_speech_active = False; self.event_tuples = []; self.drawn_lines = []
        self.spec_data_ep = np.full_like(self.spec_data, self.spec_vmin)
        self.prob_history_ep = deque(maxlen=self.spec_buffer_size); self.is_speech_active_ep = False; self.event_tuples_ep = []; self.drawn_lines_ep = []
        self.prob_history_ast = deque(maxlen=self.spec_buffer_size); self.is_speech_active_ast = False; self.event_tuples_ast = []; self.drawn_lines_ast = []
        self.prob_history_silero = deque(maxlen=self.spec_buffer_size); self.is_speech_active_silero = False; self.event_tuples_silero = []; self.drawn_lines_silero = []
        ### FIX: Añadir buffers para WebRTC
        self.prob_history_webrtc = deque(maxlen=self.spec_buffer_size); self.is_speech_active_webrtc = False; self.event_tuples_webrtc = []; self.drawn_lines_webrtc = []
        
        # Delays
        self.delay_pann = 0.0; self.deltas_pann = deque(maxlen=MAX_LAGS); self.delay_epann = 0.0; self.deltas_ep = deque(maxlen=MAX_LAGS)
        self.delay_ast = 0.0; self.deltas_ast = deque(maxlen=MAX_LAGS); self.delay_silero = 0.0; self.deltas_silero = deque(maxlen=MAX_LAGS)
        self.delay_webrtc = 0.0; self.deltas_webrtc = deque(maxlen=MAX_LAGS)

        # Shared State & Debug
        self.time_cursor = 0.0; self.running = False; self.audio_queue = queue.Queue(maxsize=100); self._residual_audio = np.array([], dtype=np.float32)
        self.last_total_samples = 0; self.animation = None; self.fig = None; self.show_debug_curve = True
        self.delay_corr_enabled = APPLY_DELAY_CORRECTION; self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.model_options = ['NONE', 'PANNs', 'E-PANNs', 'AST', 'Silero', 'WebRTC']
        ### FIX: Habilitar WebRTC en el menú
        self.disabled_models = set()
        self.selected_model_A = 'PANNs'; self.selected_model_B = 'E-PANNs'

        self._panelA_menu_open = False; self._panelB_menu_open = False
        self.energy_db_thresh = -35.0; self.prob_thresh_high = 0.50; self.energy_on_time = None; self.energy_prev_high = False
        self.prob_prev_high_pann = False; self.prob_prev_high_ep = False; self.prob_prev_high_ast = False; self.prob_prev_high_silero = False
        self.prob_prev_high_webrtc = False

        self.frame_counter = 0; self.ast_inference_interval = 80; self.last_ast_prob = 0.0

        self._load_model_and_labels(model_path, labels_path)
        self._init_audio_stream()
        self._setup_gui()

    def _robust_mean(self, arr, z=2.0):
        a = np.asarray(arr);
        if not a.size: return 0.
        med = np.median(a); mad = np.median(np.abs(a - med))
        if mad == 0: return med
        good = a[np.abs(a - med) / mad < z]
        return np.mean(good) if good.size else med

    def _add_logo(self, path, xy, zoom=1.0):
        try: self.fig.add_artist(AnnotationBbox(OffsetImage(mpimg.imread(path), zoom=zoom), xy, xycoords='figure fraction', frameon=False))
        except FileNotFoundError: print(f"⚠️  Logo not found at '{path}', skipping.")

    def _load_model_and_labels(self, model_path, labels_path):
        print(f"Loading models on device: {self.device}...")
        _, _, self.all_labels = load_csv_labels(labels_path)
        self.speech_label_indices = [self.all_labels.index(label) for label in self.SPEECH_TAGS if label in self.all_labels]
        self.model = Cnn9_GMP_64x64(len(self.all_labels))
        if os.path.exists(model_path): self.model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=False)["model"]); print(f"✓ PANNs model loaded")
        else: raise FileNotFoundError(f"PANNs model not found at {model_path}")
        self.inference = AudioModelInference(self.model, winsize=self.model_winsize, stft_hopsize=self.stft_hopsize, samplerate=self.samplerate, stft_window="hann")
        self.epanns = EPANNsVADWrapper(checkpoint="models/epanns/E-PANNs/models/checkpoint_closeto_.44.pt"); print(f"✓ E-PANNs model loaded")
        
        AST_MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
        self.ast_processor = AutoProcessor.from_pretrained(AST_MODEL_ID)
        self.ast_model = ASTForAudioClassification.from_pretrained(AST_MODEL_ID).to(self.device)
        if self.device == 'cuda': self.ast_model = self.ast_model.half()
        self.ast_model.eval()

        AST_SPEECH = {"Speech","Male speech, man speaking","Female speech, woman speaking","Child speech, kid speaking","Conversation","Narration, monologue","Babbling","Speech synthesizer","Shout","Bellow","Yell","Screaming"}
        self.ast_speech_indices = [i for i, lbl in self.ast_model.config.id2label.items() if lbl in AST_SPEECH]; print(f"✓ AST model loaded")
        
        try: self.silero_model = load_silero_vad(onnx=True); print("✓ Silero-VAD model loaded")
        except Exception as e: print(f"❌ Silero-VAD failed to load: {e}. Disabling it."); self.silero_model = None; self.disabled_models.add('Silero')
        
        ### FIX: Inicializar WebRTC VAD
        self.vad_webrtc = webrtcvad.Vad(2); print("✓ WebRTC VAD loaded")

    def _init_audio_stream(self):
        self.audiostream = MyAudioInputStream(samplerate=self.samplerate, chunk_length=self.audio_chunk_length, ringbuffer_length=self.ringbuffer_length); print("✓ Audio stream ready")

    def _setup_gui(self):
        self.fig = plt.figure(figsize=(14, 10)); self.fig.patch.set_facecolor('white')
        self.fig.suptitle("Real‑Time Speech Visualizer", fontsize=18, fontweight='bold', y=0.98)
        self.fig.subplots_adjust(left=0.07, right=0.93, bottom=0.15, top=0.95, hspace=0.3)
        self._add_logo(r'sed_demo/assets/ai4s_banner.png', (0.2, 0.95), 0.15); self._add_logo(r'sed_demo/assets/surrey_logo.png', (0.50, 0.95), 0.023); self._add_logo(r'sed_demo/assets/EPSRC_logo.png',  (0.683, 0.95), 0.10); self._add_logo(r'sed_demo/assets/CVSSP_logo.png',  (0.88, 0.945), 0.14)
        
        self.ax_pann  = self.fig.add_axes([0.16, 0.57, 0.72, 0.24]); self.im_pann  = self.ax_pann.imshow(self.spec_data, aspect='auto', origin='lower', cmap='viridis', vmin=self.spec_vmin, vmax=self.spec_vmax, interpolation='bilinear', extent=[0, self.display_duration, 0, self.display_n_mels])
        self.ax_pann.set_title('PANNs'); self.ax_pann.set_ylabel('Frequency (Hz)'); plt.setp(self.ax_pann.get_xticklabels(), visible=False)
        self.ax_pann_prob = self.ax_pann.twinx(); self.ax_pann_prob.set_ylim(0, 1); self.ax_pann_prob.set_ylabel('Prob.', color='cyan'); self.ax_pann_prob.tick_params(axis='y', colors='cyan')
        (self.line_pann_prob,) = self.ax_pann_prob.plot([], [], color='cyan', lw=1.5, alpha=.7); self.threshold_line_pann = self.ax_pann_prob.axhline(y=0.5, color='cyan', lw=1, ls=':', alpha=.9)

        self.ax_epann = self.fig.add_axes([0.16, 0.24, 0.72, 0.24]); self.im_epann = self.ax_epann.imshow(self.spec_data_ep, aspect='auto', origin='lower', cmap='viridis', vmin=self.spec_vmin, vmax=self.spec_vmax, interpolation='bilinear', extent=[0, self.display_duration, 0, self.display_n_mels])
        self.ax_epann.set_title('E-PANNs'); self.ax_epann.set_xlabel('Time (seconds)')
        self.ax_epann_prob = self.ax_epann.twinx(); self.ax_epann_prob.set_ylim(0, 1); self.ax_epann_prob.set_yticks([]); self.ax_epann_prob.tick_params(axis='y', colors='orange')
        (self.line_epann_prob,) = self.ax_epann_prob.plot([], [], color='orange', lw=1.5, alpha=.7); self.threshold_line_ep = self.ax_epann_prob.axhline(y=0.5, color='orange', lw=1, ls=':', alpha=.9)

        for ax in (self.ax_pann, self.ax_epann): ax.set_facecolor('white'); [s.set_color('black') for s in ax.spines.values()]; ax.tick_params(colors='black')
        for ax_p in (self.ax_pann_prob, self.ax_epann_prob): [s.set_color('none') for s in ax_p.spines.values()]

        self.ax_head_A = self.fig.add_axes([0.03, 0.66, 0.07, 0.025]); self.btn_head_A = Button(self.ax_head_A, f'{self.selected_model_A}  ▼', color='dimgray', hovercolor='gray'); self.btn_head_A.label.set_fontsize(8); self.btn_head_A.on_clicked(self._toggle_menu_A)
        self.ax_menu_A = self.fig.add_axes([0.03, 0.52, 0.07, 0.13]); self.rad_A = RadioButtons(self.ax_menu_A, self.model_options, active=self.model_options.index(self.selected_model_A)); self.ax_menu_A.set_visible(False); self.rad_A.on_clicked(self._choose_model_A); self.ax_menu_A.axis('off')
        for i, l in enumerate(self.rad_A.labels):
            if self.model_options[i] in self.disabled_models: l.set_color('gray')

        self.ax_head_B = self.fig.add_axes([0.03, 0.33, 0.07, 0.025]); self.btn_head_B = Button(self.ax_head_B, f'{self.selected_model_B}  ▼', color='dimgray', hovercolor='gray'); self.btn_head_B.label.set_fontsize(8); self.btn_head_B.on_clicked(self._toggle_menu_B)
        self.ax_menu_B = self.fig.add_axes([0.03, 0.19, 0.07, 0.13]); self.rad_B = RadioButtons(self.ax_menu_B, self.model_options, active=self.model_options.index(self.selected_model_B)); self.ax_menu_B.set_visible(False); self.rad_B.on_clicked(self._choose_model_B); self.ax_menu_B.axis('off')
        for i, l in enumerate(self.rad_B.labels):
            if self.model_options[i] in self.disabled_models: l.set_color('gray')

        hz = np.array([200, 500, 1000, 2000, 4000, 8000]); mel = librosa.hz_to_mel(hz, htk=True); mel_range = librosa.hz_to_mel(np.array([self.display_fmin, self.display_fmax]), htk=True); tick_pos = (mel - mel_range[0]) / (mel_range[1] - mel_range[0]) * self.display_n_mels
        self.ax_pann.set_yticks(tick_pos); self.ax_pann.set_yticklabels([f'{int(f)}' for f in hz]); self.ax_pann.set_ylim(0, self.display_n_mels)
        cbar = plt.colorbar(self.im_pann, cax=self.fig.add_axes([0.945, 0.24, 0.02, 0.57])); cbar.set_label('Power (dB)', color='black'); cbar.ax.tick_params(colors='black')
        
        self.ax_controls = self.fig.add_axes([0.07, 0.02, 0.9, 0.12]); self.ax_controls.set_facecolor('white'); self.ax_controls.axis('off')
        self.status_text = self.ax_controls.text(0.01, 0.8, 'Status: Ready', fontsize=12, transform=self.ax_controls.transAxes)
        
        state_y = [0.65, 0.50, 0.35, 0.20, 0.05] 
        self.pann_status_text   = self.ax_controls.text(0.01, state_y[0], 'PANNs: INACTIVE',  fontsize=12, color='red', transform=self.ax_controls.transAxes)
        self.epann_status_text  = self.ax_controls.text(0.01, state_y[1], 'E-PANNs: INACTIVE', fontsize=12, color='red', transform=self.ax_controls.transAxes)
        self.ast_status_text    = self.ax_controls.text(0.01, state_y[2], 'AST: INACTIVE',     fontsize=12, color='red', transform=self.ax_controls.transAxes)
        self.silero_status_text = self.ax_controls.text(0.01, state_y[3], 'Silero: INACTIVE',  fontsize=12, color='red', transform=self.ax_controls.transAxes)
        self.webrtc_status_text = self.ax_controls.text(0.01, state_y[4], 'WebRTC: INACTIVE',  fontsize=12, color='red', transform=self.ax_controls.transAxes)
        self.status_labels = {"PANNs": self.pann_status_text, "E-PANNs": self.epann_status_text, "AST": self.ast_status_text, "Silero": self.silero_status_text, "WebRTC": self.webrtc_status_text}
        for lbl in self.status_labels.values(): lbl.set_visible(False)
        
        self.prob_text = self.ax_controls.text(0.25, 0.92, 'P:0|EP:0|AST:0|S:0|W:0', fontsize=10, transform=self.ax_controls.transAxes)
        self.prob_text.set_zorder(5)
        
        self.threshold_slider = Slider(self.fig.add_axes([0.35, 0.08, 0.3, 0.03]), 'Threshold', 0.0, 1.0, valinit=0.5, valfmt='%.2f', color='cyan'); self.threshold_slider.label.set_color('black')
        self.threshold_slider.on_changed(self._on_threshold_change)
        self._on_threshold_change(self.threshold_slider.val) 
        
        self.start_stop_btn = Button(self.fig.add_axes([0.7, 0.07, 0.1, 0.04]), 'Start', color='green'); self.exit_btn = Button(self.fig.add_axes([0.85, 0.07, 0.1, 0.04]), 'Exit', color='red'); self.delay_btn = Button(self.fig.add_axes([0.57, 0.11, 0.1, 0.04]), 'Delay CORR: OFF', color='gray');
        self.delay_btn.ax.set_zorder(10) 
        self.start_stop_btn.on_clicked(self._toggle_recording); self.exit_btn.on_clicked(self._exit_app); self.delay_btn.on_clicked(self._toggle_delay_corr)
        
        # Botones de guardado
        self.save_btn = Button(self.fig.add_axes([0.28, 0.03, 0.12, 0.04]), 'Save full audio', color='skyblue')
        self.save_btn.on_clicked(self._save_snapshot)
        self.save_removed_btn = Button(self.fig.add_axes([0.43, 0.03, 0.15, 0.04]), 'Save speech removed', color='plum')
        self.save_removed_btn.on_clicked(self._save_snapshot_speech_removed)

        print("✓ GUI ready")

    def _toggle_delay_corr(self, event=None):
        self.delay_corr_enabled = not self.delay_corr_enabled; self.delay_btn.label.set_text(f'Delay CORR: {"ON" if self.delay_corr_enabled else "OFF"}'); self.delay_btn.ax.set_facecolor('green' if self.delay_corr_enabled else 'gray')
        if self.delay_corr_enabled:
            self.delay_pann  = self._robust_mean([d for d in self.deltas_pann if 0 <= d < VALID_LAG_MAX]); self.delay_epann = self._robust_mean([d for d in self.deltas_ep if 0 <= d < VALID_LAG_MAX])
            self.delay_ast   = self._robust_mean([d for d in self.deltas_ast if 0 <= d < VALID_LAG_MAX]); self.delay_silero= self._robust_mean([d for d in self.deltas_silero if 0 <= d < VALID_LAG_MAX])
            self.delay_webrtc= self._robust_mean([d for d in self.deltas_webrtc if 0 <= d < VALID_LAG_MAX])
            print(f"[Calibration] PANNs:{self.delay_pann*1000:.0f} | E-PANNs:{self.delay_epann*1000:.0f} | AST:{self.delay_ast*1000:.0f} | Silero:{self.delay_silero*1000:.0f} | WebRTC:{self.delay_webrtc*1000:.0f} (ms)")
        else: self.delay_pann = self.delay_epann = self.delay_ast = self.delay_silero = self.delay_webrtc = 0.0
        self.fig.canvas.draw_idle()
        
    def _toggle_recording(self, event=None):
        if self.running: self._stop_recording()
        else: self._start_recording()

    def _start_recording(self):
        if self.running: return
        print("\n🎤 Starting recording..."); [d.clear() for d in (self.deltas_pann, self.deltas_ep, self.deltas_ast, self.deltas_silero, self.deltas_webrtc)]
        self.running = True; self.time_cursor = 0.0; self._residual_audio = np.array([], dtype=np.float32)
        # Limpiar buffer de guardado
        self.snap_buffer.clear()
        self.energy_on_time = None; self.energy_prev_high = False; self.frame_counter = 0; self.last_event_time = {'on': -np.inf, 'off': -np.inf}
        self.prob_prev_high_pann = self.prob_prev_high_ep = self.prob_prev_high_ast = self.prob_prev_high_silero = self.prob_prev_high_webrtc = False
        if self.silero_model and hasattr(self.silero_model, "reset_states"): self.silero_model.reset_states()
        for h in (self.prob_history, self.prob_history_ep, self.prob_history_ast, self.prob_history_silero, self.prob_history_webrtc): h.clear(); h.extend([0.0] * self.spec_buffer_size)
        while not self.audio_queue.empty(): self.audio_queue.get()
        for t in (self.event_tuples, self.event_tuples_ep, self.event_tuples_ast, self.event_tuples_silero, self.event_tuples_webrtc): t.clear()
        self.last_total_samples = 0; self.is_speech_active = self.is_speech_active_ep = self.is_speech_active_ast = self.is_speech_active_silero = self.is_speech_active_webrtc = False
        self.audiostream.start(); self.processing_thread = threading.Thread(target=self._audio_processing_loop, daemon=True); self.processing_thread.start()
        self.animation = FuncAnimation(self.fig, self._update_plot, interval=1000//self.fps, blit=False, cache_frame_data=False)
        self.status_text.set_text('Status: Recording'); self.status_text.set_color('lime'); self.start_stop_btn.label.set_text('Stop'); self.start_stop_btn.ax.set_facecolor('tomato')
        print("✓ Recording started"); self.fig.canvas.draw()

    def _stop_recording(self):
        if not self.running: return
        print("\n⏹️ Stopping recording..."); self.running = False
        if self.animation: self.animation.event_source.stop(); self.animation = None
        if hasattr(self, 'processing_thread'): self.processing_thread.join(timeout=2.0)
        self.audiostream.stop()
        self.status_text.set_text('Status: Stopped'); self.status_text.set_color('orange'); self.start_stop_btn.label.set_text('Start'); self.start_stop_btn.ax.set_facecolor('green')
        
    def _exit_app(self, event=None):
        if self.running: self._stop_recording()
        self.audiostream.terminate(); plt.close(self.fig); print("\n👋 App closed.")

    def _audio_processing_loop(self):
        print("✓ Audio processing started")
        
        def _update_track(prob, state, ev, now, thr):
            new_state = prob >= thr
            if new_state != state: ev.append(('on' if new_state else 'off', now))
            return new_state

        while self.running:
            try:
                self.frame_counter += 1; full_ring_buffer = self.audiostream.read()
                n_frames = 0
                if (new_sample_count := self.audiostream.get_total_samples_written() - self.last_total_samples) > 0:
                    # Poblar buffer de guardado
                    self.snap_buffer.extend(full_ring_buffer[-new_sample_count:])
                    self._residual_audio = np.concatenate((self._residual_audio, full_ring_buffer[-new_sample_count:])); self.last_total_samples += new_sample_count
                    if len(self._residual_audio) >= self.display_n_fft:
                        mel_spec = librosa.feature.melspectrogram(y=self._residual_audio, sr=self.samplerate, n_fft=self.display_n_fft, hop_length=self.display_hop_length, n_mels=self.display_n_mels, fmin=self.display_fmin, fmax=self.display_fmax)
                        n_frames, samples_processed = mel_spec.shape[1], mel_spec.shape[1] * self.display_hop_length
                        if n_frames > 0:
                            self.audio_queue.put({'spec_display': np.clip(librosa.power_to_db(mel_spec, ref=1.0), self.spec_vmin, self.spec_vmax)})
                            frame_db = librosa.power_to_db(mel_spec.mean(axis=0), ref=1.0); self._residual_audio = self._residual_audio[samples_processed:]
                if len(self._residual_audio) > self.ringbuffer_length: self._residual_audio = self._residual_audio[-self.ringbuffer_length:]
                recent_audio = np.concatenate((np.zeros(self.samples_needed - len(full_ring_buffer)), full_ring_buffer)) if len(full_ring_buffer) < self.samples_needed else full_ring_buffer[-self.samples_needed:]
                
                thr = self.threshold_slider.val
                use_pann, use_epann, use_ast, use_silero, use_webrtc = 'PANNs' in (self.selected_model_A, self.selected_model_B), 'E-PANNs' in (self.selected_model_A, self.selected_model_B), 'AST' in (self.selected_model_A, self.selected_model_B), 'Silero' in (self.selected_model_A, self.selected_model_B), 'WebRTC' in (self.selected_model_A, self.selected_model_B)

                pann_prob = float(self.inference(recent_audio)[self.speech_label_indices].max()) if use_pann else 0.0
                self.is_speech_active = _update_track(pann_prob, self.is_speech_active, self.event_tuples, self.time_cursor, thr)
                
                with torch.no_grad(): ep_prob = float(self.epanns.audio_inference(recent_audio)[EP_SPEECH_INDICES].max()) if use_epann else 0.0
                self.is_speech_active_ep = _update_track(ep_prob, self.is_speech_active_ep, self.event_tuples_ep, self.time_cursor, thr)
                
                if use_ast:
                    if self.frame_counter % self.ast_inference_interval == 0:
                        with torch.cuda.amp.autocast(enabled=(self.device == 'cuda')):
                            wav_16k = librosa.resample(recent_audio, orig_sr=self.samplerate, target_sr=16000)
                            if wav_16k.size < 16000: wav_16k = np.pad(wav_16k, (0, 16000 - wav_16k.size))
                            else: wav_16k = wav_16k[-16000:]
                            wav_16k /= (np.abs(wav_16k).max() + 1e-9)
                            inputs = self.ast_processor(wav_16k, sampling_rate=16000, return_tensors="pt").to(self.device)
                            if self.device == 'cuda': inputs = {k: v.half() for k, v in inputs.items()}
                            with torch.no_grad(): self.last_ast_prob = torch.sigmoid(self.ast_model(**inputs).logits[0, self.ast_speech_indices]).max().item()
                    ast_prob = self.last_ast_prob
                else: ast_prob = 0.0
                self.is_speech_active_ast = _update_track(ast_prob, self.is_speech_active_ast, self.event_tuples_ast, self.time_cursor, thr)

                if use_silero and self.silero_model:
                    frame16 = librosa.resample(recent_audio[-1024:], orig_sr=self.samplerate, target_sr=16000)[-512:]
                    frame16 /= (np.abs(frame16).max() + 1e-9)
                    silero_prob = self.silero_model(torch.from_numpy(frame16).float(), 16000).item()
                else: silero_prob = 0.0
                self.is_speech_active_silero = _update_track(silero_prob, self.is_speech_active_silero, self.event_tuples_silero, self.time_cursor, thr)
                
                ### FIX: Inferencia de WebRTC
                if use_webrtc:
                    wav16 = librosa.resample(recent_audio, orig_sr=self.samplerate, target_sr=16000)
                    pcm = (wav16 * 32767).astype(np.int16).tobytes()
                    frame_len_ms = 30; frame_len_samples = 480 # 30ms a 16kHz
                    num_frames = len(pcm) // (frame_len_samples * 2)
                    voiced = sum(1 for i in range(num_frames) if self.vad_webrtc.is_speech(pcm[i*frame_len_samples*2:(i+1)*frame_len_samples*2], 16000))
                    webrtc_prob = voiced / max(1, num_frames)
                else: webrtc_prob = 0.0
                self.is_speech_active_webrtc = _update_track(webrtc_prob, self.is_speech_active_webrtc, self.event_tuples_webrtc, self.time_cursor, thr)

                if n_frames > 0:
                    self.time_cursor += n_frames * self.spec_time_resolution
                    for h,p in [(self.prob_history, pann_prob), (self.prob_history_ep, ep_prob), (self.prob_history_ast, ast_prob), (self.prob_history_silero, silero_prob), (self.prob_history_webrtc, webrtc_prob)]: h.extend([p] * n_frames)
                    for i in range(n_frames):
                        t_frame = self.time_cursor - (n_frames - i) * self.spec_time_resolution
                        if (energy_high := frame_db[i] > self.energy_db_thresh) and not self.energy_prev_high: self.energy_on_time = t_frame
                        self.energy_prev_high = energy_high
                        def check_vad_onset(prob, prev_high, deltas):
                            is_high = prob >= self.prob_thresh_high
                            if is_high and not prev_high and self.energy_on_time is not None:
                                if 0.0 <= (lag := t_frame - self.energy_on_time) < VALID_LAG_MAX: deltas.append(lag)
                            return is_high
                        self.prob_prev_high_pann = check_vad_onset(pann_prob, self.prob_prev_high_pann, self.deltas_pann)
                        self.prob_prev_high_ep = check_vad_onset(ep_prob, self.prob_prev_high_ep, self.deltas_ep)
                        self.prob_prev_high_ast = check_vad_onset(ast_prob, self.prob_prev_high_ast, self.deltas_ast)
                        self.prob_prev_high_silero = check_vad_onset(silero_prob, self.prob_prev_high_silero, self.deltas_silero)
                        self.prob_prev_high_webrtc = check_vad_onset(webrtc_prob, self.prob_prev_high_webrtc, self.deltas_webrtc)
                time.sleep(0.005)
            except Exception as e: print(f"[audio-thread] Warning → {e}"); continue

    def _get_curve_data(self, label): return {'PANNs':(self.prob_history,'cyan'), 'E-PANNs':(self.prob_history_ep,'orange'), 'AST':(self.prob_history_ast,'yellow'), 'Silero':(self.prob_history_silero,'springgreen'), 'WebRTC':(self.prob_history_webrtc, 'magenta'), 'NONE':([], 'gray')}.get(label, ([], 'gray'))

    def _update_plot(self, frame):
        try:
            while not self.audio_queue.empty():
                data = self.audio_queue.get_nowait()
                if 'spec_display' in data and (chunk_size := data['spec_display'].shape[1]) > 0:
                    for spec in (self.spec_data, self.spec_data_ep): spec[:] = np.roll(spec, -chunk_size, 1); spec[:, -chunk_size:] = data['spec_display'][:, -chunk_size:]
            self._recompute_events()
            if self.prob_history: self.prob_text.set_text(f'P:{self.prob_history[-1]:.2f}|EP:{self.prob_history_ep[-1]:.2f}|AST:{self.prob_history_ast[-1]:.2f}|S:{self.prob_history_silero[-1]:.2f}|W:{self.prob_history_webrtc[-1]:.2f}')
            
            delay_map = {"PANNs": self.delay_pann, "E-PANNs": self.delay_epann, "AST": self.delay_ast, "Silero": self.delay_silero, "WebRTC": self.delay_webrtc}
            for line, ax_p, thr, model in [(self.line_pann_prob, self.ax_pann_prob, self.threshold_line_pann, self.selected_model_A), (self.line_epann_prob, self.ax_epann_prob, self.threshold_line_ep, self.selected_model_B)]:
                curv, col = self._get_curve_data(model); line.set_color(col); ax_p.set_ylabel('Prob.', color=col); ax_p.tick_params(axis='y', colors=col); thr.set_color(col)
                if len(curv): x_data = np.linspace(0, self.display_duration, self.spec_buffer_size); line.set_data(x_data[-len(curv):] - (delay_map.get(model,0.0) if self.delay_corr_enabled else 0), list(curv))
                else: line.set_data([], [])
            self.im_pann.set_data(self.spec_data); self.im_epann.set_data(self.spec_data_ep)
        except Exception as e: print(f"Error updating plot: {e}")

    def _on_threshold_change(self, val):
        self.prob_thresh_high = val
        for line in (self.threshold_line_pann, self.threshold_line_ep): line.set_ydata([val])
        self.fig.canvas.draw_idle()

    def _recompute_events(self):
        m_data = {"PANNs": (self.event_tuples, self.is_speech_active), "E-PANNs": (self.event_tuples_ep, self.is_speech_active_ep), "AST": (self.event_tuples_ast, self.is_speech_active_ast), "Silero": (self.event_tuples_silero, self.is_speech_active_silero), "WebRTC": (self.event_tuples_webrtc, self.is_speech_active_webrtc), "NONE": ([], False)}
        drawing_map = {"PANNs": (self.drawn_lines, self.pann_status_text), "E-PANNs": (self.drawn_lines_ep, self.epann_status_text), "AST": (self.drawn_lines_ast, self.ast_status_text), "Silero": (self.drawn_lines_silero, self.silero_status_text), "WebRTC": (self.drawn_lines_webrtc, self.webrtc_status_text), "NONE": ([], None)}
        
        for lbl in self.status_labels.values(): lbl.set_visible(False)
        
        for model_name, ax in [(self.selected_model_A, self.ax_pann), (self.selected_model_B, self.ax_epann)]:
            drawn_list, status_text = drawing_map.get(model_name, ([], None))
            tuples, is_active = m_data.get(model_name, ([], False))
            if status_text: self._draw_events(ax, drawn_list, list(tuples), is_active, status_text, model_name)
            else: # Limpiar panel si es NONE
                for ln in drawn_list: ln.remove()
                drawn_list.clear()

    def _draw_events(self, ax, drawn_list, tuples, is_active, status_text, model):
        for ln in drawn_list: ln.remove();
        drawn_list.clear()
        if tuples and tuples[-1][0] == 'on' and not is_active: tuples.pop()
        
        delay = {"PANNs": self.delay_pann, "E-PANNs": self.delay_epann, "AST": self.delay_ast, "Silero": self.delay_silero, "WebRTC": self.delay_webrtc}.get(model, 0.0)
        start = self.time_cursor - self.display_duration
        for k, t in tuples:
            if t > start: drawn_list.append(ax.axvline((t - (delay if self.delay_corr_enabled else 0)) - start, color='lime' if k == 'on' else 'red', lw=2, ls='--', alpha=0.8))
        
        status_text.set_visible(True)
        status_text.set_text(f'{model}: {"ACTIVE" if is_active else "INACTIVE"}'); status_text.set_color('lime' if is_active else 'red')

    def _clear_markers(self):
        for lst in [self.drawn_lines, self.drawn_lines_ep, self.drawn_lines_ast, self.drawn_lines_silero, self.drawn_lines_webrtc]:
            for l in lst: l.remove()
            lst.clear()

    def _toggle_menu_A(self, e=None): self._panelA_menu_open = not self._panelA_menu_open; self.ax_menu_A.set_visible(self._panelA_menu_open); self.fig.canvas.draw_idle()
    def _toggle_menu_B(self, e=None): self._panelB_menu_open = not self._panelB_menu_open; self.ax_menu_B.set_visible(self._panelB_menu_open); self.fig.canvas.draw_idle()
    def _choose_model_A(self, l): self._choose_model(l, True)
    def _choose_model_B(self, l): self._choose_model(l, False)

    def _choose_model(self, label, is_A):
        rad, current = (self.rad_A, self.selected_model_A) if is_A else (self.rad_B, self.selected_model_B)
        if label in self.disabled_models: rad.set_active(self.model_options.index(current)); return
        
        for name, lbl in self.status_labels.items(): lbl.set_visible(False)
        
        if is_A: 
            self.selected_model_A = label; self.ax_pann.set_title(label); self.btn_head_A.label.set_text(f'{label}  ▼'); self._toggle_menu_A()
        else: 
            self.selected_model_B = label; self.ax_epann.set_title(label); self.btn_head_B.label.set_text(f'{label}  ▼'); self._toggle_menu_B()
        
        if label != 'NONE': self.status_labels[label].set_visible(True); self.status_labels[label].set_text(f'{label}: INACTIVE'); self.status_labels[label].set_color('red')
        self._clear_markers(); self._update_plot(None); self.fig.canvas.draw_idle()

    def _dump_annotations(self, path):
        def to_pairs(lst): return [(on[1], off[1]) for on, off in zip(lst[::2], lst[1::2])]
        ev = {"PANNs": self.event_tuples, "E-PANNs": self.event_tuples_ep, "AST": self.event_tuples_ast, "Silero": self.event_tuples_silero, "WebRTC": self.event_tuples_webrtc}
        data = {"threshold": self.threshold_slider.val}
        if self.selected_model_A != 'NONE': data[self.selected_model_A] = to_pairs(ev.get(self.selected_model_A, []))
        if self.selected_model_B not in ('NONE', self.selected_model_A): data[self.selected_model_B] = to_pairs(ev.get(self.selected_model_B, []))
        with open(path, 'w', encoding='utf-8') as f: json.dump(data, f, indent=2)

    def _save_snapshot(self, _event=None):
        audio = np.asarray(self.snap_buffer, dtype=np.float32)
        if not audio.size: print("⚠️ Nada que guardar."); return
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        wav_name = f"vad_{ts}.wav"
        json_name = f"vad_{ts}.json"
        audio_i16 = (np.clip(audio, -1, 1) * 32767).astype('<i2')
        with wave.open(wav_name, 'wb') as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(self.samplerate)
            wf.writeframes(audio_i16.tobytes())
        self._dump_annotations(json_name)
        print(f"✓ Guardado {wav_name} y {json_name}")

    def _save_snapshot_speech_removed(self, _evt=None):
        audio = np.asarray(self.snap_buffer, dtype=np.float32).copy()
        if not audio.size: print("⚠️ Nada que guardar."); return
        t0 = self.time_cursor - 10.0
        ev = {"PANNs": self.event_tuples, "E-PANNs": self.event_tuples_ep, "AST": self.event_tuples_ast, "Silero": self.event_tuples_silero, "WebRTC": self.event_tuples_webrtc}[self.selected_model_A]
        pairs = [(on[1], off[1]) for on, off in zip(ev[::2], ev[1::2])]
        for on_t, off_t in pairs:
            i0 = max(0, int((on_t  - t0) * self.samplerate))
            i1 = max(0, int((off_t - t0) * self.samplerate))
            audio[i0:i1] = 0.0
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        wav_name = f"vad_{ts}_removed.wav"
        json_name = f"vad_{ts}.json"
        audio_i16 = (np.clip(audio, -1, 1) * 32767).astype('<i2')
        with wave.open(wav_name, 'wb') as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(self.samplerate)
            wf.writeframes(audio_i16.tobytes())
        self._dump_annotations(json_name)
        print(f"✓ Guardado {wav_name} y {json_name} (speech removido)")

    def run(self):
        print("\n" + "="*60 + "\nReal-Time VAD Comparison\n" + "="*60); print("  - Select models from the dropdown menus (left).\n  - Press 'Start' and speak to see real-time VAD.\n  - Press 'Delay CORR' to apply the measured compensation.\n")
        self.ax_pann.set_xlim(0, self.display_duration); self.ax_epann.set_xlim(0, self.display_duration); plt.show()

def main():
    try: SpeechDetectionApp().run()
    except KeyboardInterrupt: print("\n\nStopped by user.")
    except Exception as e: print(f"\n❌ An unhandled error occurred: {e}"); import traceback; traceback.print_exc()

if __name__ == "__main__": main()