#!/usr/bin/env python3
"""
Example Usage of Emotion Prediction API

This script demonstrates various ways to use the emotion prediction system
programmatically in your own Python code.
"""

import sys
from pathlib import Path

# Add test directory to path
sys.path.append(str(Path(__file__).parent))

from predict import EmotionPredictor


def example_1_single_prediction():
    """Example 1: Predict emotion for a single file"""
    print("\n" + "="*60)
    print("Example 1: Single File Prediction")
    print("="*60)
    
    # Initialize predictor
    predictor = EmotionPredictor(
        model_path='../selected/final_best_vit/best_model.pth',
        model_type='best_vit'
    )
    
    # Predict
    result = predictor.predict('../dataset/DEAM/MEMD_audio/10.mp3')
    
    # Access results
    print(f"\nFile: {result['file']}")
    print(f"Valence: {result['valence']:.4f}")
    print(f"Arousal: {result['arousal']:.4f}")
    print(f"Valence (0-1): {result['valence_normalized']:.4f}")
    print(f"Arousal (0-1): {result['arousal_normalized']:.4f}")


def example_2_batch_prediction():
    """Example 2: Predict emotions for multiple files"""
    print("\n" + "="*60)
    print("Example 2: Batch Prediction")
    print("="*60)
    
    # Initialize predictor
    predictor = EmotionPredictor(
        model_path='../selected/final_best_vit/best_model.pth',
        model_type='best_vit'
    )
    
    # List of files
    audio_files = [
        '../dataset/DEAM/MEMD_audio/10.mp3',
        '../dataset/DEAM/MEMD_audio/100.mp3',
        '../dataset/DEAM/MEMD_audio/1000.mp3',
    ]
    
    # Predict batch
    results = predictor.predict_batch(audio_files)
    
    # Process results
    print(f"\nProcessed {len(results)} files:")
    for result in results:
        if 'error' not in result:
            print(f"  {result['file']}: V={result['valence']:+.3f}, A={result['arousal']:+.3f}")


def example_3_custom_preprocessing():
    """Example 3: Use custom audio preprocessing"""
    print("\n" + "="*60)
    print("Example 3: Custom Preprocessing")
    print("="*60)
    
    from audio_preprocessor import AudioPreprocessor
    
    # Initialize preprocessor with custom settings
    preprocessor = AudioPreprocessor(
        sample_rate=22050,
        duration=30,
        n_mels=128,
        image_size=224
    )
    
    # Process audio
    audio_path = '../dataset/DEAM/MEMD_audio/10.mp3'
    image_tensor, mel_spec = preprocessor.preprocess(audio_path)
    
    print(f"\nPreprocessed {Path(audio_path).name}")
    print(f"  Image tensor shape: {image_tensor.shape}")
    print(f"  Mel spectrogram shape: {mel_spec.shape}")


def example_4_analyze_dataset():
    """Example 4: Analyze a dataset of songs"""
    print("\n" + "="*60)
    print("Example 4: Dataset Analysis")
    print("="*60)
    
    from pathlib import Path
    import pandas as pd
    
    # Initialize predictor
    predictor = EmotionPredictor(
        model_path='../selected/final_best_vit/best_model.pth',
        model_type='best_vit'
    )
    
    # Find audio files
    audio_dir = Path('../dataset/DEAM/MEMD_audio')
    audio_files = sorted(list(audio_dir.glob('*.mp3')))[:10]  # First 10 files
    
    # Predict
    print(f"\nAnalyzing {len(audio_files)} songs...")
    results = predictor.predict_batch([str(f) for f in audio_files])
    
    # Create DataFrame
    df = pd.DataFrame([
        {
            'file': r['file'],
            'valence': r['valence'],
            'arousal': r['arousal']
        }
        for r in results if 'error' not in r
    ])
    
    # Statistics
    print(f"\nDataset Statistics:")
    print(f"  Mean Valence: {df['valence'].mean():+.4f}")
    print(f"  Mean Arousal: {df['arousal'].mean():+.4f}")
    print(f"  Valence Std:  {df['valence'].std():.4f}")
    print(f"  Arousal Std:  {df['arousal'].std():.4f}")
    
    # Find extremes
    print(f"\nMost Positive: {df.loc[df['valence'].idxmax(), 'file']}")
    print(f"Most Negative: {df.loc[df['valence'].idxmin(), 'file']}")
    print(f"Most Energetic: {df.loc[df['arousal'].idxmax(), 'file']}")
    print(f"Most Calm: {df.loc[df['arousal'].idxmin(), 'file']}")


def example_5_compare_models():
    """Example 5: Compare different model architectures"""
    print("\n" + "="*60)
    print("Example 5: Model Comparison")
    print("="*60)
    
    import time
    
    audio_path = '../dataset/DEAM/MEMD_audio/10.mp3'
    models = [
        ('best_vit', '../selected/final_best_vit/best_model.pth'),
        ('mobile_vit', '../selected/final_best_vit/mobile_vit_student.pth'),
    ]
    
    print(f"\nComparing models on: {Path(audio_path).name}\n")
    
    for model_type, model_path in models:
        if not Path(model_path).exists():
            print(f"⚠️  {model_type}: Model not found, skipping...")
            continue
        
        # Initialize predictor
        predictor = EmotionPredictor(
            model_path=model_path,
            model_type=model_type
        )
        
        # Measure inference time
        start_time = time.time()
        result = predictor.predict(audio_path)
        inference_time = time.time() - start_time
        
        # Print results
        print(f"{model_type.upper()}:")
        print(f"  Valence: {result['valence']:+.4f}")
        print(f"  Arousal: {result['arousal']:+.4f}")
        print(f"  Inference Time: {inference_time:.3f}s")
        print()


def example_6_save_results():
    """Example 6: Save predictions to file"""
    print("\n" + "="*60)
    print("Example 6: Save Results to File")
    print("="*60)
    
    from pathlib import Path
    import json
    import csv
    
    # Initialize predictor
    predictor = EmotionPredictor(
        model_path='../selected/final_best_vit/best_model.pth',
        model_type='best_vit'
    )
    
    # Get predictions
    audio_files = [
        '../dataset/DEAM/MEMD_audio/10.mp3',
        '../dataset/DEAM/MEMD_audio/100.mp3',
        '../dataset/DEAM/MEMD_audio/1000.mp3',
    ]
    results = predictor.predict_batch(audio_files)
    
    # Save as JSON
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    
    json_path = output_dir / 'predictions_example.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved JSON: {json_path}")
    
    # Save as CSV
    csv_path = output_dir / 'predictions_example.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['file', 'valence', 'arousal', 
                                                'valence_normalized', 'arousal_normalized'])
        writer.writeheader()
        for result in results:
            if 'error' not in result:
                writer.writerow(result)
    print(f"✓ Saved CSV: {csv_path}")


def main():
    """Run all examples"""
    print("\n" + "="*60)
    print("🎵 EMOTION PREDICTION API EXAMPLES")
    print("="*60)
    
    try:
        example_1_single_prediction()
        example_2_batch_prediction()
        example_3_custom_preprocessing()
        example_4_analyze_dataset()
        example_5_compare_models()
        example_6_save_results()
        
        print("\n" + "="*60)
        print("✅ All examples completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
