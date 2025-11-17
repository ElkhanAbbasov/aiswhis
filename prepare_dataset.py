"""
Dataset preparation utilities for XTTS-v2 Turkish fine-tuning
Helps you prepare your audio data in LJSpeech format
"""

import os
import wave
import json
from pathlib import Path
from typing import List, Tuple, Optional


def check_audio_format(wav_path: str) -> dict:
    """
    Check if audio file meets XTTS requirements
    
    Requirements:
    - 16-bit PCM
    - 22.05 kHz or 24 kHz sample rate
    - Mono or stereo (mono preferred)
    """
    try:
        with wave.open(wav_path, 'rb') as wav:
            info = {
                'channels': wav.getnchannels(),
                'sample_width': wav.getsampwidth(),
                'framerate': wav.getframerate(),
                'n_frames': wav.getnframes(),
                'duration': wav.getnframes() / wav.getframerate(),
            }
            
            # Check requirements
            info['valid'] = True
            info['issues'] = []
            
            if info['sample_width'] != 2:  # 16-bit = 2 bytes
                info['valid'] = False
                info['issues'].append(f"Sample width should be 16-bit (2 bytes), got {info['sample_width']}")
            
            if info['framerate'] not in [22050, 24000]:
                info['valid'] = False
                info['issues'].append(f"Sample rate should be 22050 or 24000 Hz, got {info['framerate']}")
            
            if info['channels'] > 2:
                info['valid'] = False
                info['issues'].append(f"Too many channels: {info['channels']} (should be 1 or 2)")
            
            return info
    except Exception as e:
        return {
            'valid': False,
            'error': str(e)
        }


def validate_dataset(dataset_root: str, metadata_file: str = "metadata.txt") -> dict:
    """
    Validate dataset structure and audio files
    
    Expected structure:
    dataset_root/
        wavs/
            0001.wav
            0002.wav
            ...
        metadata.txt
    """
    dataset_root = Path(dataset_root)
    wavs_dir = dataset_root / "wavs"
    metadata_path = dataset_root / metadata_file
    
    results = {
        'valid': True,
        'total_files': 0,
        'valid_files': 0,
        'invalid_files': [],
        'total_duration': 0.0,
        'issues': []
    }
    
    # Check directory structure
    if not dataset_root.exists():
        results['valid'] = False
        results['issues'].append(f"Dataset root not found: {dataset_root}")
        return results
    
    if not wavs_dir.exists():
        results['valid'] = False
        results['issues'].append(f"'wavs' directory not found in {dataset_root}")
        return results
    
    if not metadata_path.exists():
        results['valid'] = False
        results['issues'].append(f"metadata.txt not found in {dataset_root}")
        return results
    
    # Read metadata
    metadata_entries = []
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split('|')
                if len(parts) != 3:
                    results['issues'].append(
                        f"Line {line_num}: Expected 3 columns, got {len(parts)}: {line[:50]}"
                    )
                    continue
                
                metadata_entries.append({
                    'id': parts[0],
                    'text': parts[1],
                    'normalized_text': parts[2],
                    'line_num': line_num
                })
    except Exception as e:
        results['valid'] = False
        results['issues'].append(f"Error reading metadata: {e}")
        return results
    
    results['total_files'] = len(metadata_entries)
    
    # Validate audio files
    for entry in metadata_entries:
        wav_path = wavs_dir / f"{entry['id']}.wav"
        
        if not wav_path.exists():
            results['invalid_files'].append({
                'id': entry['id'],
                'issue': 'File not found'
            })
            continue
        
        audio_info = check_audio_format(str(wav_path))
        
        if not audio_info.get('valid', False):
            results['invalid_files'].append({
                'id': entry['id'],
                'issues': audio_info.get('issues', [audio_info.get('error', 'Unknown error')])
            })
        else:
            results['valid_files'] += 1
            results['total_duration'] += audio_info['duration']
    
    # Summary
    if results['invalid_files']:
        results['valid'] = False
        results['issues'].append(f"Found {len(results['invalid_files'])} invalid audio files")
    
    # Calculate statistics
    results['duration_hours'] = results['total_duration'] / 3600
    results['avg_duration'] = results['total_duration'] / max(results['valid_files'], 1)
    
    return results


def create_sample_metadata(output_path: str, wav_dir: str):
    """
    Create a sample metadata.txt from wav files in a directory
    Note: You'll need to fill in the actual transcripts!
    """
    wav_dir = Path(wav_dir)
    wav_files = sorted(wav_dir.glob("*.wav"))
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, wav_file in enumerate(wav_files, 1):
            file_id = wav_file.stem
            # Placeholder text - REPLACE WITH ACTUAL TRANSCRIPTS!
            f.write(f"{file_id}|[TRANSCRIPT HERE]|[TRANSCRIPT HERE]\n")
    
    print(f"Created sample metadata at: {output_path}")
    print(f"Found {len(wav_files)} wav files")
    print("⚠ WARNING: You must replace '[TRANSCRIPT HERE]' with actual Turkish transcripts!")


def print_validation_report(results: dict):
    """Print a formatted validation report"""
    print("\n" + "=" * 60)
    print("DATASET VALIDATION REPORT")
    print("=" * 60)
    
    if results['valid']:
        print("✓ Dataset is valid and ready for training!")
    else:
        print("✗ Dataset has issues that need to be fixed")
    
    print(f"\nFiles: {results['valid_files']}/{results['total_files']} valid")
    print(f"Total duration: {results['duration_hours']:.2f} hours")
    print(f"Average clip duration: {results['avg_duration']:.2f} seconds")
    
    if results['issues']:
        print("\n⚠ Issues found:")
        for issue in results['issues']:
            print(f"  - {issue}")
    
    if results['invalid_files']:
        print(f"\n✗ Invalid files ({len(results['invalid_files'])}):")
        for item in results['invalid_files'][:10]:  # Show first 10
            print(f"  - {item['id']}: {item.get('issue', item.get('issues', 'Unknown'))}")
        
        if len(results['invalid_files']) > 10:
            print(f"  ... and {len(results['invalid_files']) - 10} more")
    
    print("=" * 60 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate XTTS Turkish dataset")
    parser.add_argument("--dataset-root", type=str, required=True, help="Path to dataset root")
    parser.add_argument("--create-sample", action="store_true", help="Create sample metadata.txt")
    parser.add_argument("--wav-dir", type=str, help="Wav directory for sample creation")
    
    args = parser.parse_args()
    
    if args.create_sample:
        if not args.wav_dir:
            print("Error: --wav-dir required for --create-sample")
        else:
            output = os.path.join(args.dataset_root, "metadata.txt")
            create_sample_metadata(output, args.wav_dir)
    else:
        results = validate_dataset(args.dataset_root)
        print_validation_report(results)
        
        # Save detailed report
        report_path = os.path.join(args.dataset_root, "validation_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Detailed report saved to: {report_path}")
