@echo off
echo === VAD Demo – instalador Windows ===

python -c "import sys;sys.exit(0 if sys.version_info>=(3,9) else 1)" || (
  echo [ERROR] Instala Python 3.9+; saliendo…
  pause & exit /b 1
)

python -m venv vad_env
call "%~dp0vad_env\Scripts\activate.bat"
python -m pip install -U pip setuptools wheel

REM ====== dependencias ======
(
echo librosa>=0.9.2
echo soundfile>=0.12.1
echo PyAudio==0.2.14
echo torch>=2.1.0
echo torchaudio>=2.1.0
echo torchlibrosa==0.0.4
echo torchsummary==1.5.1
echo transformers>=4.20.0
echo numpy>=1.23.0
echo scipy>=1.11.0
echo numba>=0.56
echo matplotlib>=3.8.0
echo Pillow>=10.0.0
echo omegaconf>=2.3.0
echo silero-vad>=5.0.0
echo webrtcvad-wheels>=2.0.14
echo onnxruntime>=1.22.0
) > _reqs.txt
pip install -r _reqs.txt
del _reqs.txt
REM ====== /dependencias ======

REM ====== assets que faltaban ======
powershell -c "mkdir models\panns, models\epanns\E-PANNs\models -ea 0"
powershell -c "Invoke-WebRequest -Uri 'http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv' -OutFile 'models\\panns\\class_labels_indices.csv'"
powershell -c "Invoke-WebRequest -Uri 'https://zenodo.org/records/7939403/files/checkpoint_closeto_.44.pt?download=1' -OutFile 'models\\epanns\\E-PANNs\\models\\checkpoint_closeto_.44.pt'"
REM ====== /assets ======

python download_models.py

echo.&echo [OK] Instalada. Ejecuta ^.^\run_app.bat
pause
