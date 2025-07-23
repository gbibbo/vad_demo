@echo off
echo 🎤 Starting VAD Demo...

if not exist "vad_env\Scripts\activate.bat" (
    echo ❌ Virtual environment not found. Please run install_windows.bat first.
    pause
    exit /b 1
)

echo 🔌 Activating virtual environment...
call vad_env\Scripts\activate.bat

echo 🎵 IMPORTANT: Grant microphone permissions when prompted!
echo.

echo 🚀 Launching application...
python speech_detection_app.py

pause
