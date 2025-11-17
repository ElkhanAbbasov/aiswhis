# XTTS-v2 Turkish Fine-tuning Pipeline

Complete pipeline for fine-tuning XTTS-v2 on Turkish audio dataset (~2 hours) using an L4 GPU on Google Colab.

## 📋 Overview

This pipeline allows you to:
- Fine-tune XTTS-v2's GPT encoder on your Turkish voice dataset
- Maintain multilingual capability while adapting to your specific voice
- Deploy the model for Turkish text-to-speech in your AI agents

## 🎯 Requirements

### Hardware
- **Google Colab with L4 GPU** (recommended)
- Runtime → Change runtime type → GPU → L4
- ~15-20GB GPU memory for training

### Dataset
- **~2 hours** of Turkish audio (minimum 1 hour, more is better)
- **Format**: 16-bit PCM WAV, 22.05 kHz or 24 kHz, mono
- **Structure**: LJSpeech-style with `metadata.txt`

### Dataset Structure
```
xtts_tr_dataset/
├── wavs/
│   ├── 0001.wav
│   ├── 0002.wav
│   └── ...
└── metadata.txt
```

### metadata.txt Format
```text
0001|Bu bir örnek cümledir.|Bu bir örnek cümledir.
0002|İkinci cümle burada.|İkinci cümle burada.
...
```

Format: `utterance_id|transcript|normalized_transcript`

## 🚀 Quick Start

### Step 1: Prepare Your Dataset

1. **Validate your dataset**:
```bash
python prepare_dataset.py --dataset-root /path/to/your/dataset
```

2. **Check audio quality**:
- All files should be 16-bit PCM
- Sample rate: 22.05 kHz or 24 kHz
- Duration: 3-10 seconds per clip (optimal)

3. **Upload to Google Drive**:
```
/MyDrive/xtts_tr_dataset/
├── wavs/
└── metadata.txt
```

### Step 2: Set Up Google Colab

1. **Open Colab**: https://colab.research.google.com/
2. **Select L4 GPU**: Runtime → Change runtime type → L4 GPU
3. **Upload files**: Upload `colab_setup.py` and `train_gpt_xtts_tr.py`

### Step 3: Install Dependencies

```python
# Cell 1: Check GPU
!nvidia-smi

# Cell 2: Run setup
!python colab_setup.py
```

### Step 4: Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')

# Verify dataset
import os
dataset_root = "/content/drive/MyDrive/xtts_tr_dataset"
print("Wav files:", len([f for f in os.listdir(f"{dataset_root}/wavs") if f.endswith(".wav")]))
print("Metadata exists:", os.path.isfile(f"{dataset_root}/metadata.txt"))
```

### Step 5: Start Training

```python
import os

# Set environment variables
os.environ["XTTS_TR_DATASET_ROOT"] = "/content/drive/MyDrive/xtts_tr_dataset"
os.environ["XTTS_TR_METADATA_FILE"] = "/content/drive/MyDrive/xtts_tr_dataset/metadata.txt"

# Start training
!python train_gpt_xtts_tr.py
```

### Step 6: Monitor Training

Training will output:
- Loss metrics every 50 steps
- Evaluation results periodically
- Checkpoints every 10,000 steps
- TensorBoard logs in `./run_tr_tr/training/`

**Expected training time**: 4-8 hours on L4 GPU for 2h dataset

### Step 7: Test Your Model

```python
!python inference_turkish.py \
    --text "Merhaba, ben senin Türkçe konuşan yapay zeka asistanınım." \
    --speaker-wav /content/drive/MyDrive/xtts_tr_dataset/wavs/0001.wav \
    --output sample_output.wav

# Listen to output
from IPython.display import Audio
Audio("sample_output.wav")
```

## 📁 Project Structure

```
aiswhis/
├── train_gpt_xtts_tr.py        # Main training script
├── colab_setup.py               # Dependency installation
├── inference_turkish.py         # Inference script
├── prepare_dataset.py           # Dataset validation utilities
├── README.md                    # This file
└── run_tr_tr/                   # Training outputs (created during training)
    └── training/
        └── XTTSv2_Turkish_FT-*/
            ├── best_model.pth
            ├── config.json
            └── checkpoints/
```

## 🔧 Configuration

### Training Parameters (in `train_gpt_xtts_tr.py`)

```python
BATCH_SIZE = 4              # Reduce if OOM (out of memory)
GRAD_ACUMM_STEPS = 64       # Effective batch = BATCH_SIZE × GRAD_ACUMM_STEPS
lr = 5e-6                   # Learning rate
```

### Audio Settings

```python
sample_rate = 22050         # Input audio sample rate
output_sample_rate = 24000  # Output audio sample rate
max_wav_length = 255995     # ~11.6 seconds max
```

## 💡 Tips for Best Results

### Dataset Quality
- **Clean audio**: Minimal background noise
- **Consistent speaker**: Single speaker works best
- **Natural speech**: Conversational tone, not monotone
- **Proper transcripts**: Accurate Turkish text with correct punctuation

### Training
- **Monitor loss**: Should decrease steadily
- **Checkpoint often**: Training can be interrupted
- **Evaluate periodically**: Listen to test samples
- **Use early stopping**: Avoid overfitting on small datasets

### Inference
- **Good reference audio**: Use 3-6 second clear clips
- **Text length**: Keep sentences reasonably short
- **Temperature**: Adjust for creativity vs. stability

## 🐛 Troubleshooting

### OOM (Out of Memory)
```python
# Reduce batch size
BATCH_SIZE = 2  # or even 1
```

### Slow training
- Verify L4 GPU is active: `!nvidia-smi`
- Check if using CPU: Look for "CUDA available: True" in logs

### Poor quality output
- **More data**: 2h is minimum, 4-5h is better
- **Better audio quality**: Clean recordings
- **Longer training**: Let it train longer
- **Check loss**: Should be < 2.0 for good quality

### Metadata errors
```python
# Validate format
!python prepare_dataset.py --dataset-root /your/dataset
```

### Model loading errors
```python
# Add safe globals (should be done in colab_setup.py)
import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig

torch.serialization.add_safe_globals([
    XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs
])
```

## 📊 Performance Expectations

| Dataset Size | Training Time (L4) | Quality |
|-------------|-------------------|---------|
| 1 hour      | 2-4 hours         | Fair    |
| 2 hours     | 4-8 hours         | Good    |
| 4 hours     | 8-16 hours        | Very Good |
| 6+ hours    | 16+ hours         | Excellent |

## 🔌 Integration with AI Agents

### Python Integration

```python
from inference_turkish import generate_speech

# Generate Turkish speech
audio_file = generate_speech(
    text="Merhaba, nasıl yardımcı olabilirim?",
    speaker_wav="reference_voice.wav",
    output_file="response.wav"
)
```

### API Server Example

```python
from flask import Flask, request, send_file
from inference_turkish import generate_speech

app = Flask(__name__)

@app.route('/tts', methods=['POST'])
def text_to_speech():
    text = request.json['text']
    speaker_wav = request.json.get('speaker_wav', 'default_voice.wav')
    
    output = generate_speech(text, speaker_wav, "temp_output.wav")
    return send_file(output, mimetype='audio/wav')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

## 📚 Resources

- [Coqui TTS Documentation](https://docs.coqui.ai/)
- [XTTS-v2 Paper](https://arxiv.org/abs/2406.04904)
- [LJSpeech Dataset Format](https://keithito.com/LJ-Speech-Dataset/)

## 📝 License

This project uses:
- **Coqui TTS** (Mozilla Public License 2.0)
- **XTTS-v2** (Coqui Public Model License)

## 🤝 Contributing

Contributions welcome! Please:
1. Test on your Turkish dataset
2. Report issues with dataset details
3. Share improvements and optimizations

## ✨ Acknowledgments

- Coqui AI team for XTTS-v2
- Turkish TTS community
- Google Colab for L4 GPU access

---

**Need help?** Open an issue with:
- Your dataset specifications
- Training configuration
- Error messages and logs
- GPU and environment details
