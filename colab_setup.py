# ====================================================================
# SIMPLE ONE-CELL INSTALLATION FOR GOOGLE COLAB
# Copy this entire cell and run it in Google Colab
# ====================================================================

# Step 1: Uninstall conflicting packages
print("🧹 Cleaning up...")
!pip uninstall -y TTS pandas gruut -q

# Step 2: Install dependencies in correct order
print("📦 Installing compatible Pandas...")
!pip install -q 'pandas>=1.4,<2.0'

print("📦 Installing Gruut 2.2.3...")
!pip install -q 'gruut[de,es,fr]==2.2.3'

print("📦 Installing language processing tools...")
!pip install -q anyascii unidecode pysbd num2words g2pkk jamo hangul-romanize pypinyin bangla bnnumerizer bnunicodenormalizer

print("📦 Installing audio processing...")
!pip install -q librosa soundfile encodec

print("📦 Installing TTS 0.22.0...")
!pip install -q TTS==0.22.0 --no-deps

print("📦 Installing remaining dependencies...")
!pip install -q coqpit fsspec aiohttp packaging tqdm trainer tensorboard matplotlib Pillow mecab-python3 unidic-lite unidic

# Step 3: Verify installation
print("\n" + "="*60)
print("🔍 VERIFYING INSTALLATION")
print("="*60)

try:
    import TTS
    print(f"✅ TTS: {TTS.__version__}")
    
    import pandas as pd
    print(f"✅ Pandas: {pd.__version__}")
    
    import gruut
    print(f"✅ Gruut: {gruut.__version__}")
    
    import torch
    print(f"✅ PyTorch: {torch.__version__}")
    print(f"✅ CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    
    # Configure safe globals for model loading
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts, XttsAudioConfig, XttsArgs
    from TTS.config.shared_configs import BaseDatasetConfig
    
    torch.serialization.add_safe_globals([
        XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs
    ])
    
    print(f"✅ XTTS modules imported successfully")
    print("\n" + "="*60)
    print("✨ INSTALLATION COMPLETE! ✨")
    print("="*60)
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    print("\n⚠️ If you see this error, try restarting runtime and running again")

print("\n📋 Next Steps:")
print("1. Mount Google Drive")
print("2. Verify your dataset")  
print("3. Start training!")