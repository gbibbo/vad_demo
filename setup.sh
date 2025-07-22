#!/bin/bash

echo "=== Speech Detection App Setup Script ==="
echo "This script will help you set up the environment for the real-time speech detection application."
echo

# Check if we're in a conda environment
if [[ "$CONDA_DEFAULT_ENV" != "" ]]; then
    echo "✓ Conda environment detected: $CONDA_DEFAULT_ENV"
else
    echo "⚠ Warning: No conda environment detected. It's recommended to create one first:"
    echo "  conda create -n speech_detection python=3.8"
    echo "  conda activate speech_detection"
    echo
fi

# Install system dependencies for PyAudio
echo "Installing system dependencies for PyAudio..."
if command -v apt-get &> /dev/null; then
    echo "Detected apt package manager (Ubuntu/Debian)"
    sudo apt-get update
    sudo apt-get install -y portaudio19-dev python3-pyaudio alsa-utils
elif command -v yum &> /dev/null; then
    echo "Detected yum package manager (RHEL/CentOS)"
    sudo yum install -y portaudio-devel alsa-lib-devel
elif command -v brew &> /dev/null; then
    echo "Detected Homebrew (macOS)"
    brew install portaudio
else
    echo "⚠ Could not detect package manager. You may need to install portaudio manually."
fi

echo

# Install Python dependencies
echo "Installing Python dependencies..."

# Try conda first for PyAudio
if command -v conda &> /dev/null; then
    echo "Installing PyAudio via conda..."
    conda install -c anaconda pyaudio -y
else
    echo "Installing PyAudio via pip..."
    pip install PyAudio
fi

# Install other requirements
echo "Installing other Python packages..."
pip install torch>=1.7.0 torchlibrosa==0.0.4 librosa>=0.8.0 numpy>=1.19.0 scipy>=1.5.0 soundfile>=0.10.3 matplotlib>=3.3.0 Pillow>=8.4.0 omegaconf>=2.1.1 numba>=0.45

echo

# Download the model
echo "Downloading pre-trained model..."
MODEL_FILE="Cnn9_GMP_64x64_300000_iterations_mAP=0.37.pth"

if [ ! -f "$MODEL_FILE" ]; then
    echo "Downloading model from Zenodo..."
    wget "https://zenodo.org/record/3576599/files/Cnn9_GMP_64x64_300000_iterations_mAP%3D0.37.pth?download=1" -O "$MODEL_FILE"
    
    if [ $? -eq 0 ]; then
        echo "✓ Model downloaded successfully"
    else
        echo "✗ Failed to download model. Please download manually:"
        echo "wget \"https://zenodo.org/record/3576599/files/Cnn9_GMP_64x64_300000_iterations_mAP%3D0.37.pth?download=1\" -O \"$MODEL_FILE\""
    fi
else
    echo "✓ Model file already exists"
fi

echo

# Test installations
echo "Testing installations..."

# Test PyAudio
python3 -c "import pyaudio; print('✓ PyAudio installed successfully')" 2>/dev/null || echo "✗ PyAudio installation failed"

# Test other dependencies
python3 -c "import torch; print('✓ PyTorch installed successfully')" 2>/dev/null || echo "✗ PyTorch installation failed"
python3 -c "import librosa; print('✓ Librosa installed successfully')" 2>/dev/null || echo "✗ Librosa installation failed"
python3 -c "import matplotlib; print('✓ Matplotlib installed successfully')" 2>/dev/null || echo "✗ Matplotlib installation failed"

echo

# Final instructions
echo "=== Setup Complete ==="
echo
echo "To run the speech detection app:"
echo "  python3 speech_detection_app.py"
echo
echo "Make sure you have:"
echo "  1. A working microphone connected"
echo "  2. Audio permissions enabled"
echo "  3. The sed_demo folder in the same directory"
echo
echo "Controls:"
echo "  - Click 'Start' to begin recording"
echo "  - Use the threshold slider to adjust sensitivity"
echo "  - Green lines indicate speech onset"
echo "  - Red lines indicate speech offset"
echo
echo "If you encounter issues with audio, try:"
echo "  - Check microphone permissions"
echo "  - Run: pulseaudio --start (Linux)"
echo "  - Verify audio devices: python3 -c \"import pyaudio; p=pyaudio.PyAudio(); [print(p.get_device_info_by_index(i)) for i in range(p.get_device_count())]\""