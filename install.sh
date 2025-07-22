#!/bin/bash

# Real-Time Speech Detection & Privacy-Preserving Audio Processing
# Automatic Installation Script

set -e  # Exit on any error

echo "🎤 Real-Time Speech Detection & Privacy-Preserving Audio Processing"
echo "=================================================================="
echo "This script will set up the complete environment for VAD demo."
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in a conda environment
if [[ "$CONDA_DEFAULT_ENV" != "" ]]; then
    print_success "Conda environment detected: $CONDA_DEFAULT_ENV"
else
    print_warning "No conda environment detected."
    echo "It's recommended to create one first:"
    echo "  conda create -n vad_demo python=3.8"
    echo "  conda activate vad_demo"
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check Python version
print_status "Checking Python version..."
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
REQUIRED_VERSION="3.8"

if python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
    print_success "Python $PYTHON_VERSION is compatible"
else
    print_error "Python $PYTHON_VERSION is not compatible. Required: $REQUIRED_VERSION+"
    exit 1
fi

# Install system dependencies
print_status "Installing system dependencies..."

if command -v apt-get &> /dev/null; then
    print_status "Detected apt package manager (Ubuntu/Debian)"
    sudo apt-get update -qq
    sudo apt-get install -y portaudio19-dev python3-pyaudio alsa-utils ffmpeg
elif command -v yum &> /dev/null; then
    print_status "Detected yum package manager (RHEL/CentOS)"
    sudo yum install -y portaudio-devel alsa-lib-devel ffmpeg
elif command -v dnf &> /dev/null; then
    print_status "Detected dnf package manager (Fedora)"
    sudo dnf install -y portaudio-devel alsa-lib-devel ffmpeg
elif command -v brew &> /dev/null; then
    print_status "Detected Homebrew (macOS)"
    brew install portaudio ffmpeg
elif command -v pacman &> /dev/null; then
    print_status "Detected pacman package manager (Arch Linux)"
    sudo pacman -S --noconfirm portaudio alsa-utils ffmpeg
else
    print_warning "Could not detect package manager. You may need to install portaudio manually."
    print_warning "Required packages: portaudio, alsa-utils (Linux), ffmpeg"
fi

# Install Python dependencies
print_status "Installing Python dependencies..."

# Try conda first for certain packages
if command -v conda &> /dev/null; then
    print_status "Installing audio packages via conda..."
    conda install -c anaconda pyaudio -y || print_warning "Failed to install PyAudio via conda, will try pip"
    conda install -c conda-forge librosa -y || print_warning "Failed to install librosa via conda, will try pip"
fi

# Install requirements
print_status "Installing Python packages from requirements.txt..."
pip install --upgrade pip
pip install -r requirements.txt

# Install additional dependencies that might conflict
print_status "Installing additional audio dependencies..."
pip install webrtcvad-wheels
pip install silero-vad

# Download models
print_status "Downloading pre-trained models..."
python3 download_models.py

# Test installations
print_status "Testing installations..."

test_package() {
    local package=$1
    local import_name=${2:-$package}
    if python3 -c "import $import_name" 2>/dev/null; then
        print_success "$package installed successfully"
        return 0
    else
        print_error "$package installation failed"
        return 1
    fi
}

# Test core packages
test_package "PyAudio" "pyaudio"
test_package "PyTorch" "torch"
test_package "Librosa" "librosa"
test_package "Matplotlib" "matplotlib"
test_package "Transformers" "transformers"
test_package "WebRTC VAD" "webrtcvad"
test_package "Silero VAD" "silero_vad"
test_package "SoundFile" "soundfile"

# Test audio system
print_status "Testing audio system..."
if python3 -c "
import pyaudio
p = pyaudio.PyAudio()
device_count = p.get_device_count()
print(f'Found {device_count} audio devices')
p.terminate()
" 2>/dev/null; then
    print_success "Audio system is working"
else
    print_warning "Audio system test failed. You may need to check audio permissions."
fi

# Check required files
print_status "Checking required files..."
required_files=(
    "speech_detection_app.py"
    "sed_demo/models.py"
    "sed_demo/utils.py"
    "sed_demo/inference.py"
    "sed_demo/audio_loop.py"
    "sed_demo/assets/audioset_labels.csv"
    "src/wrappers/vad_epanns.py"
)

missing_files=()
for file in "${required_files[@]}"; do
    if [[ -f "$file" ]]; then
        print_success "Found $file"
    else
        print_error "Missing $file"
        missing_files+=("$file")
    fi
done

if [[ ${#missing_files[@]} -gt 0 ]]; then
    print_error "Missing required files. Please ensure you have cloned the complete repository."
    exit 1
fi

# Final test
print_status "Running final system test..."
if python3 -c "
import torch
import librosa
import matplotlib
import pyaudio
from sed_demo.models import Cnn9_GMP_64x64
from src.wrappers.vad_epanns import EPANNsVADWrapper
print('✅ All imports successful')
" 2>/dev/null; then
    print_success "All systems ready!"
else
    print_error "System test failed. Please check the error messages above."
    exit 1
fi

echo ""
echo "🎉 Installation Complete!"
echo "========================"
echo ""
echo "To run the speech detection demo:"
echo "  python3 speech_detection_app.py"
echo ""
echo "📋 Quick checklist:"
echo "  ✅ Python 3.8+ installed"
echo "  ✅ System audio dependencies installed"
echo "  ✅ Python packages installed"
echo "  ✅ Pre-trained models downloaded"
echo "  ✅ Required files present"
echo ""
echo "🎛️ Usage tips:"
echo "  • Make sure your microphone is connected and working"
echo "  • Grant audio permissions when prompted"
echo "  • Use the dropdown menus to select models for comparison"
echo "  • Adjust the threshold slider for optimal detection"
echo "  • Click 'Start' to begin real-time processing"
echo ""
echo "❓ If you encounter issues:"
echo "  • Check microphone permissions"
echo "  • Verify audio device: python3 -c \"import pyaudio; p=pyaudio.PyAudio(); [print(f'{i}: {p.get_device_info_by_index(i)[\\\"name\\\"]}') for i in range(p.get_device_count())]\""
echo "  • Check our troubleshooting guide: https://github.com/gbibbo/vad_demo/issues"
echo ""
echo "Happy speech detection! 🗣️✨"