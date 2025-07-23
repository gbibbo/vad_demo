@echo off
echo 🎤 VAD Demo - Windows Installation
echo =================================

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python not found. Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo ✅ Python found
echo 📦 Creating virtual environment...
python -m venv vad_env

echo 🔌 Activating virtual environment...
call vad_env\Scripts\activate.bat

echo ⬆️ Upgrading pip...
python -m pip install --upgrade pip

echo 🎵 Installing audio dependencies...
pip install pipwin
pipwin install pyaudio

echo 📚 Installing Python packages...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install webrtcvad-wheels silero-vad librosa soundfile matplotlib numpy scipy numba transformers omegaconf Pillow torchlibrosa onnxruntime

echo 📥 Downloading models...
python download_models.py

echo 🎉 Installation complete!
echo.
echo To run the application in PowerShell, use: .\run_app.bat
pause
