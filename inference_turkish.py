"""
Inference script for fine-tuned Turkish XTTS-v2 model
Use this to generate Turkish speech from your trained model
"""

import glob
import os
import argparse
from pathlib import Path

def find_latest_model(run_root="./run_tr_tr/training"):
    """Find the latest trained model checkpoint"""
    runs = sorted(glob.glob(os.path.join(run_root, "XTTSv2_Turkish_FT-*")))
    
    if not runs:
        raise FileNotFoundError(f"No trained models found in {run_root}")
    
    model_dir = runs[-1]
    model_path = os.path.join(model_dir, "best_model.pth")
    config_path = os.path.join(model_dir, "config.json")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    return model_path, config_path


def generate_speech(
    text,
    speaker_wav,
    output_file="output_turkish.wav",
    model_path=None,
    config_path=None,
    language="tr",
    use_gpu=True
):
    """
    Generate Turkish speech from text
    
    Args:
        text: Turkish text to synthesize
        speaker_wav: Path to reference speaker audio (3-6 seconds)
        output_file: Output wav file path
        model_path: Path to trained model (auto-detected if None)
        config_path: Path to model config (auto-detected if None)
        language: Language code (default: "tr" for Turkish)
        use_gpu: Whether to use GPU for inference
    """
    from TTS.api import TTS
    
    # Auto-detect model if not provided
    if model_path is None or config_path is None:
        print("Auto-detecting latest model...")
        model_path, config_path = find_latest_model()
    
    print(f"Using model: {model_path}")
    print(f"Using config: {config_path}")
    
    # Initialize TTS
    tts = TTS(
        model_path=model_path,
        config_path=config_path,
        gpu=use_gpu,
    )
    
    # Generate speech
    print(f"\nGenerating speech for: '{text}'")
    print(f"Using speaker reference: {speaker_wav}")
    
    tts.tts_to_file(
        text=text,
        file_path=output_file,
        speaker_wav=speaker_wav,
        language=language,
    )
    
    print(f"✓ Generated: {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(description="Turkish XTTS-v2 Inference")
    parser.add_argument("--text", type=str, required=True, help="Turkish text to synthesize")
    parser.add_argument("--speaker-wav", type=str, required=True, help="Reference speaker audio file")
    parser.add_argument("--output", type=str, default="output_turkish.wav", help="Output audio file")
    parser.add_argument("--model-path", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--config-path", type=str, default=None, help="Path to config file")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU")
    
    args = parser.parse_args()
    
    # Verify speaker wav exists
    if not os.path.exists(args.speaker_wav):
        print(f"Error: Speaker wav file not found: {args.speaker_wav}")
        return
    
    # Generate speech
    generate_speech(
        text=args.text,
        speaker_wav=args.speaker_wav,
        output_file=args.output,
        model_path=args.model_path,
        config_path=args.config_path,
        use_gpu=not args.no_gpu
    )


if __name__ == "__main__":
    # Example usage if run without arguments
    import sys
    if len(sys.argv) == 1:
        print("Example usage:")
        print('  python inference_turkish.py --text "Merhaba dünya" --speaker-wav reference.wav')
        print("\nOr use in Python:")
        print('  from inference_turkish import generate_speech')
        print('  generate_speech("Merhaba dünya", "reference.wav", "output.wav")')
    else:
        main()
