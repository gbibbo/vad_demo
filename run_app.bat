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
echo If you don't see permission dialog, check Windows Privacy Settings:
echo Settings ^> Privacy ^& Security ^> Microphone ^> Allow apps to access microphone
echo.

echo 🚀 Launching application...
python speech_detection_app.py

pause
