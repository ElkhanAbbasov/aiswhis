"""
Colab Setup Script for XTTS-v2 Turkish Fine-tuning
This script handles the initial setup including PyTorch installation,
TTS library installation, and safe globals configuration.
"""

import subprocess
import sys

def install_dependencies():
    """Install PyTorch, TTS, and required dependencies"""
    print("=" * 60)
    print("Installing PyTorch with CUDA 12.1 support...")
    print("=" * 60)
    
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--upgrade", "pip"
    ])
    
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "torch==2.3.1", "torchaudio==2.3.1",
        "--index-url", "https://download.pytorch.org/whl/cu121"
    ])
    
    print("\n" + "=" * 60)
    print("Installing Coqui TTS and Trainer...")
    print("=" * 60)
    
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "numpy<2.0.0",
        "transformers==4.35.2",
        "TTS==0.22.0",
        "coqui-tts-trainer",
    ])
    
    print("\n" + "=" * 60)
    print("Installation complete!")
    print("=" * 60)


def configure_safe_globals():
    """Configure torch serialization safe globals for XTTS"""
    print("\n" + "=" * 60)
    print("Configuring safe globals for XTTS...")
    print("=" * 60)
    
    try:
        import torch
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
        from TTS.config.shared_configs import BaseDatasetConfig
        
        torch.serialization.add_safe_globals([
            XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs
        ])
        
        print("✓ Safe globals configured successfully")
    except Exception as e:
        print(f"⚠ Warning: Could not configure safe globals: {e}")
        print("This may not be an issue depending on your PyTorch version")


def check_gpu():
    """Check if GPU is available"""
    print("\n" + "=" * 60)
    print("Checking GPU availability...")
    print("=" * 60)
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print("⚠ No GPU detected. Training will be very slow on CPU.")
    except ImportError:
        print("⚠ PyTorch not yet installed. Run install_dependencies() first.")


if __name__ == "__main__":
    print("XTTS-v2 Turkish Fine-tuning Setup\n")
    
    # Install dependencies
    install_dependencies()
    
    # Configure safe globals
    configure_safe_globals()
    
    # Check GPU
    check_gpu()
    
    print("\n" + "=" * 60)
    print("Setup complete! You can now proceed with training.")
    print("=" * 60)
