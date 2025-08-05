@echo off
REM === VAD Demo ? instalador Windows ===
python -c "import sys;exit(0) if sys.version_info>=(3,9) else exit(1)" || (
  echo [ERROR] Necesitas Python 3.9+ & exit /b 1
)

python -m venv vad_env
call "%~dp0vad_env\Scripts\activate.bat"
python -m pip install -U pip setuptools wheel

REM ---- dependencias binarias (una sola l?nea) ----
pip install --no-cache-dir --only-binary=:all: ^
  numpy==2.1.2 transformers==4.54.1 silero-vad==5.1.2 webrtcvad-wheels==2.0.14 ^
  PyAudio==0.2.14 torchlibrosa==0.0.4 torchsummary==1.5.1 ^
  torch>=2.1.0 torchaudio>=2.1.0 librosa>=0.9.2 soundfile>=0.12.1 ^
  scipy>=1.11.0 numba>=0.56 matplotlib>=3.8.0 Pillow>=10.0.0 ^
  omegaconf>=2.3.0 onnxruntime>=1.22.0

REM ---- assets ----
powershell -c "mkdir models\panns, models\epanns\E-PANNs\models -ea 0"
powershell -c "Invoke-WebRequest -Uri 'http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv' -OutFile 'models\\panns\\class_labels_indices.csv'"
powershell -c "Invoke-WebRequest -Uri 'https://zenodo.org/records/7939403/files/checkpoint_closeto_.44.pt?download=1' -OutFile 'models\\epanns\\E-PANNs\\models\\checkpoint_closeto_.44.pt'"

python download_models.py
echo.&echo [OK] Instalada. Ejecuta .\run_app.bat
pause
