#!/usr/bin/env python3
"""
Automatic Model Download Script for VAD Demo
"""

import os
import sys
import urllib.request
import urllib.error
from pathlib import Path
from tqdm import tqdm
import json

class ModelDownloader:
    def __init__(self):
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        
        self.models = {
            "panns_cnn9": {
                "url": "https://zenodo.org/record/3576599/files/Cnn9_GMP_64x64_300000_iterations_mAP%3D0.37.pth?download=1",
                "filename": "Cnn9_GMP_64x64_300000_iterations_mAP=0.37.pth",
                "size_mb": 142,
                "description": "PANNs CNN9 model for audio tagging"
            }
        }

    def download_with_progress(self, url, filepath):
        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            class ProgressHook:
                def __init__(self):
                    self.pbar = None
                
                def __call__(self, block_num, block_size, total_size):
                    if self.pbar is None:
                        self.pbar = tqdm(total=total_size, unit='B', unit_scale=True, 
                                       desc=f"Downloading {filepath.name}")
                    
                    downloaded = block_num * block_size
                    if downloaded < total_size:
                        self.pbar.update(block_size)
                    else:
                        self.pbar.close()

            urllib.request.urlretrieve(url, filepath, ProgressHook())
            return True
            
        except Exception as e:
            print(f"❌ Error downloading {filepath.name}: {e}")
            return False

    def verify_file(self, filepath, expected_size_mb=None):
        if not filepath.exists():
            return False
        
        if expected_size_mb:
            actual_size_mb = filepath.stat().st_size / (1024 * 1024)
            if abs(actual_size_mb - expected_size_mb) > expected_size_mb * 0.1:
                print(f"⚠️  File size mismatch for {filepath.name}")
                return False
        
        return True

    def download_model(self, model_key, model_info):
        print(f"\n📥 Downloading {model_info['description']}...")
        
        filepath = Path(model_info['filename'])
        
        if self.verify_file(filepath, model_info.get('size_mb')):
            print(f"✅ {filepath.name} already exists and appears valid")
            return True
        
        print(f"🌐 Downloading from: {model_info['url']}")
        success = self.download_with_progress(model_info['url'], filepath)
        
        if success and self.verify_file(filepath, model_info.get('size_mb')):
            print(f"✅ Successfully downloaded {filepath.name}")
            return True
        else:
            print(f"❌ Failed to download or verify {filepath.name}")
            return False

    def check_library_models(self):
        print("\n🔍 Checking library-based models...")
        
        try:
            print("📦 Testing AST (Audio Spectrogram Transformer)...")
            from transformers import AutoProcessor, ASTForAudioClassification
            
            model_id = "MIT/ast-finetuned-audioset-10-10-0.4593"
            processor = AutoProcessor.from_pretrained(model_id)
            model = ASTForAudioClassification.from_pretrained(model_id)
            print("✅ AST model downloaded and ready")
        except Exception as e:
            print(f"❌ AST model test failed: {e}")
        
        try:
            print("📦 Testing Silero VAD...")
            from silero_vad import load_silero_vad
            model = load_silero_vad(onnx=True)
            print("✅ Silero VAD model downloaded and ready")
        except Exception as e:
            print(f"❌ Silero VAD test failed: {e}")
        
        try:
            print("📦 Testing WebRTC VAD...")
            import webrtcvad
            vad = webrtcvad.Vad(2)
            print("✅ WebRTC VAD ready")
        except Exception as e:
            print(f"❌ WebRTC VAD test failed: {e}")

    def run(self):
        print("🤖 VAD Demo Model Downloader")
        print("=" * 40)
        
        success_count = 0
        for model_key, model_info in self.models.items():
            if self.download_model(model_key, model_info):
                success_count += 1
        
        self.check_library_models()
        
        if success_count == len(self.models):
            print("\n🎉 All models downloaded successfully!")
        else:
            print(f"\n⚠️  Some downloads failed.")
        
        return success_count == len(self.models)

def main():
    if not Path("speech_detection_app.py").exists():
        print("❌ Error: Please run this script from the vad_demo repository root directory.")
        sys.exit(1)
    
    downloader = ModelDownloader()
    success = downloader.run()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
