#!/usr/bin/env python3
"""
Automatic Model Download Script for VAD Demo
"""

import os
import sys
import urllib.request
from pathlib import Path
import json

def download_panns_model():
    """Download PANNs model if not exists."""
    model_file = "Cnn9_GMP_64x64_300000_iterations_mAP=0.37.pth"
    
    if Path(model_file).exists():
        print(f"✅ {model_file} already exists")
        return True
    
    print(f"📥 Downloading {model_file}...")
    url = "https://zenodo.org/record/3576599/files/Cnn9_GMP_64x64_300000_iterations_mAP%3D0.37.pth?download=1"
    
    try:
        urllib.request.urlretrieve(url, model_file)
        print(f"✅ Successfully downloaded {model_file}")
        return True
    except Exception as e:
        print(f"❌ Failed to download {model_file}: {e}")
        return False

def test_library_models():
    """Test if library-based models can be loaded."""
    print("\n🔍 Testing library-based models...")
    
    # Test transformers (AST)
    try:
        print("📦 Testing AST model...")
        from transformers import AutoProcessor, ASTForAudioClassification
        model_id = "MIT/ast-finetuned-audioset-10-10-0.4593"
        processor = AutoProcessor.from_pretrained(model_id)
        print("✅ AST model ready")
    except Exception as e:
        print(f"⚠️ AST model test failed: {e}")
    
    # Test Silero VAD  
    try:
        print("📦 Testing Silero VAD...")
        from silero_vad import load_silero_vad
        model = load_silero_vad(onnx=True)
        print("✅ Silero VAD ready")
    except Exception as e:
        print(f"⚠️ Silero VAD test failed: {e}")
    
    # Test WebRTC VAD
    try:
        print("📦 Testing WebRTC VAD...")
        import webrtcvad
        vad = webrtcvad.Vad(2)
        print("✅ WebRTC VAD ready")
    except Exception as e:
        print(f"⚠️ WebRTC VAD test failed: {e}")

def main():
    print("🤖 VAD Demo Model Downloader")
    print("=" * 40)
    
    if not Path("speech_detection_app.py").exists():
        print("❌ Error: Run this script from the vad_demo root directory.")
        sys.exit(1)
    
    # Download PANNs model
    success = download_panns_model()
    
    # Test library models
    test_library_models()
    
    if success:
        print("\n🎉 Model setup complete!")
    else:
        print("\n⚠️ Some models failed to download.")

if __name__ == "__main__":
    main()
