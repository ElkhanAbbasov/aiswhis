# Google Colab Notebook Guide

Copy and paste these cells into your Google Colab notebook in order.

## Cell 1: Check GPU

```python
# Verify L4 GPU is active
!nvidia-smi
```

Expected output should show "NVIDIA L4" with ~24GB memory.

## Cell 2: Install Dependencies

```python
# Install PyTorch with CUDA 12.1
!pip install --upgrade pip
!pip install "torch==2.3.1" "torchaudio==2.3.1" --index-url https://download.pytorch.org/whl/cu121

# Install Coqui TTS and trainer
!pip install "TTS==0.22.0" "coqui-tts-trainer"

# Configure safe globals
import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig

torch.serialization.add_safe_globals([
    XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs
])

print("✓ Setup complete!")
```

## Cell 3: Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')

# Set your dataset path
DATASET_ROOT = "/content/drive/MyDrive/xtts_tr_dataset"
METADATA_FILE = f"{DATASET_ROOT}/metadata.txt"

# Quick check
import os
print("Wav count:", len([f for f in os.listdir(f"{DATASET_ROOT}/wavs") if f.endswith(".wav")]))
print("Metadata exists:", os.path.isfile(METADATA_FILE))
```

## Cell 4: Upload Training Script

Upload `train_gpt_xtts_tr.py` from your local machine to Colab using the file browser.

Or create it directly:

```python
%%writefile train_gpt_xtts_tr.py
# [Paste full content of train_gpt_xtts_tr.py here]
```

## Cell 5: Start Training

```python
import os

# Set environment variables
os.environ["XTTS_TR_DATASET_ROOT"] = "/content/drive/MyDrive/xtts_tr_dataset"
os.environ["XTTS_TR_METADATA_FILE"] = "/content/drive/MyDrive/xtts_tr_dataset/metadata.txt"

# Start training
!python train_gpt_xtts_tr.py
```

## Cell 6: Monitor Training (Optional - Run in separate cell while training)

```python
# Load TensorBoard
%load_ext tensorboard
%tensorboard --logdir ./run_tr_tr/training/
```

## Cell 7: Test Inference After Training

```python
import glob, os
from TTS.api import TTS

# Find latest run dir
run_root = "/content/run_tr_tr/training"
runs = sorted(glob.glob(os.path.join(run_root, "XTTSv2_Turkish_FT-*")))
assert runs, "No runs found"
model_dir = runs[-1]

model_path = os.path.join(model_dir, "best_model.pth")
config_path = os.path.join(model_dir, "config.json")

print("Using model:", model_path)

# Initialize TTS
tts = TTS(
    model_path=model_path,
    config_path=config_path,
    gpu=True,
)

# Generate sample
speaker_wav = "/content/drive/MyDrive/xtts_tr_dataset/wavs/0001.wav"

tts.tts_to_file(
    text="Merhaba, ben senin Türkçe konuşan yapay zeka asistanınım.",
    file_path="sample_tr.wav",
    speaker_wav=speaker_wav,
    language="tr",
)

print("✓ Generated: sample_tr.wav")
```

## Cell 8: Listen to Output

```python
from IPython.display import Audio
Audio("sample_tr.wav")
```

## Cell 9: Generate More Samples

```python
test_sentences = [
    "Bugün hava çok güzel.",
    "Nasıl yardımcı olabilirim?",
    "Türkçe konuşmak için ince ayarlanmış bir modelim.",
    "Yapay zeka teknolojisi hızla gelişiyor.",
]

for i, text in enumerate(test_sentences):
    output_file = f"test_{i+1}.wav"
    tts.tts_to_file(
        text=text,
        file_path=output_file,
        speaker_wav=speaker_wav,
        language="tr",
    )
    print(f"✓ Generated: {output_file}")
    display(Audio(output_file))
```

## Cell 10: Download Trained Model

```python
# Zip the trained model
!zip -r trained_model.zip {model_dir}

# Download to your computer
from google.colab import files
files.download('trained_model.zip')
```

## Tips

1. **Save checkpoints to Drive**: Modify `OUT_PATH` in training script to save to Drive
2. **Resume training**: Set `restore_path` to your checkpoint
3. **Monitor loss**: Loss should decrease to ~1.5-2.5 for good quality
4. **Adjust batch size**: If OOM, reduce `BATCH_SIZE` in script

## Expected Timeline

- Setup: 5-10 minutes
- Dataset upload: 10-30 minutes (depending on size)
- Training: 4-8 hours for 2h dataset
- Testing: 2-5 minutes

## Troubleshooting

### GPU Not Active
```python
# Force reconnect to L4
from google.colab import runtime
runtime.unassign()
```

### Training Too Slow
```python
# Verify GPU usage
import torch
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
```

### OOM Error
Reduce batch size in `train_gpt_xtts_tr.py`:
```python
BATCH_SIZE = 2  # or 1
```

---

**Pro tip**: Keep the Colab tab active and check progress every hour. Training can take 4-8 hours.
