#!/usr/bin/env python3
"""
Quick Test Script for Emotion Prediction

This script provides a quick way to test the emotion prediction system
with a few sample audio files from the DEAM dataset.

Usage:
    python quick_test.py
"""

import sys
from pathlib import Path
from predict import EmotionPredictor, format_emotion_output

def main():
    # Define paths
    base_dir = Path(__file__).parent.parent
    model_dir = base_dir / 'selected' / 'final_best_vit'
    audio_dir = base_dir / 'dataset' / 'DEAM' / 'MEMD_audio'
    
    # Sample audio files to test
    sample_files = ['10.mp3', '100.mp3', '1000.mp3', '500.mp3', '50.mp3']
    
    print("="*60)
    print("🎵 EMOTION PREDICTION QUICK TEST")
    print("="*60)
    
    # Check if directories exist
    if not model_dir.exists():
        print(f"\n❌ Model directory not found: {model_dir}")
        print("   Please ensure models are extracted to the correct location.")
        sys.exit(1)
    
    if not audio_dir.exists():
        print(f"\n❌ Audio directory not found: {audio_dir}")
        print("   Please ensure DEAM dataset is available.")
        sys.exit(1)
    
    # Check for model file
    model_path = model_dir / 'best_model.pth'
    if not model_path.exists():
        print(f"\n❌ Model file not found: {model_path}")
        print("\nAvailable files in model directory:")
        for f in model_dir.glob('*'):
            print(f"   - {f.name}")
        sys.exit(1)
    
    # Find available sample files
    available_files = []
    for sample in sample_files:
        audio_path = audio_dir / sample
        if audio_path.exists():
            available_files.append(str(audio_path))
    
    if not available_files:
        print(f"\n❌ No sample audio files found in: {audio_dir}")
        print("   Looking for:", ", ".join(sample_files))
        sys.exit(1)
    
    print(f"\n✅ Found {len(available_files)} sample audio files")
    print(f"✅ Model ready: {model_path.name}")
    
    # Initialize predictor
    print("\n🔄 Loading model...")
    try:
        predictor = EmotionPredictor(
            model_path=str(model_path),
            model_type='best_vit'
        )
    except Exception as e:
        print(f"\n❌ Failed to load model: {e}")
        sys.exit(1)
    
    # Process samples
    print(f"\n🎵 Testing emotion prediction on {len(available_files)} samples...")
    print("="*60)
    
    results = []
    for audio_path in available_files:
        try:
            result = predictor.predict(audio_path)
            results.append(result)
            print(format_emotion_output(result))
        except Exception as e:
            print(f"\n❌ Error processing {Path(audio_path).name}: {e}")
    
    # Summary
    if results:
        print("\n" + "="*60)
        print("📊 QUICK TEST SUMMARY")
        print("="*60)
        print(f"\nSuccessfully processed: {len(results)}/{len(available_files)} files")
        
        valences = [r['valence'] for r in results]
        arousals = [r['arousal'] for r in results]
        
        print(f"\nAverage Valence: {sum(valences)/len(valences):+.4f}")
        print(f"Average Arousal: {sum(arousals)/len(arousals):+.4f}")
        
        print("\n✅ Quick test complete!")
        print("\nNext steps:")
        print("  1. Try single file: python predict.py --audio_file path/to/song.mp3")
        print("  2. Try batch mode: python batch_predict.py --audio_dir ../dataset/DEAM/MEMD_audio --n_samples 10")
        print("  3. Read README.md for more options")
    else:
        print("\n❌ No successful predictions")
        print("   Check error messages above for details")
    
    print("="*60)


if __name__ == '__main__':
    main()
