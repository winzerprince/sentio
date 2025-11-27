#!/usr/bin/env python3
"""
Batch Emotion Prediction from Multiple Audio Files

This script processes multiple audio files and generates a report of
emotion predictions (valence and arousal).

Usage:
    python batch_predict.py --audio_dir ../dataset/DEAM/MEMD_audio --n_samples 10
    python batch_predict.py --audio_files song1.mp3 song2.mp3 song3.mp3
"""

import os
import sys
import argparse
import torch
import pandas as pd
from pathlib import Path
from datetime import datetime
import json

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from predict import EmotionPredictor, interpret_valence, interpret_arousal, interpret_emotion


def find_audio_files(directory, extensions=['.mp3', '.wav', '.flac', '.ogg'], limit=None):
    """Find audio files in directory"""
    audio_files = []
    
    directory = Path(directory)
    if not directory.exists():
        raise ValueError(f"Directory not found: {directory}")
    
    for ext in extensions:
        audio_files.extend(directory.glob(f'*{ext}'))
    
    # Sort by filename
    audio_files = sorted(audio_files)
    
    # Limit if specified
    if limit:
        audio_files = audio_files[:limit]
    
    return [str(f) for f in audio_files]


def save_results(results, output_dir, format='csv'):
    """Save prediction results to file"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create DataFrame
    df_data = []
    for result in results:
        if 'error' not in result:
            df_data.append({
                'file': result['file'],
                'valence': result['valence'],
                'arousal': result['arousal'],
                'valence_normalized': result['valence_normalized'],
                'arousal_normalized': result['arousal_normalized'],
                'valence_interpretation': interpret_valence(result['valence']),
                'arousal_interpretation': interpret_arousal(result['arousal']),
                'emotion': interpret_emotion(result['valence'], result['arousal'])
            })
        else:
            df_data.append({
                'file': result['file'],
                'error': result['error']
            })
    
    df = pd.DataFrame(df_data)
    
    # Save CSV
    if format in ['csv', 'both']:
        csv_path = output_dir / f'predictions_{timestamp}.csv'
        df.to_csv(csv_path, index=False)
        print(f"\n📄 Results saved to: {csv_path}")
    
    # Save JSON
    if format in ['json', 'both']:
        json_path = output_dir / f'predictions_{timestamp}.json'
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"📄 Results saved to: {json_path}")
    
    return df


def print_summary_statistics(results):
    """Print summary statistics of predictions"""
    # Filter out errors
    valid_results = [r for r in results if 'error' not in r]
    
    if not valid_results:
        print("\n❌ No valid predictions to summarize")
        return
    
    # Calculate statistics
    valences = [r['valence'] for r in valid_results]
    arousals = [r['arousal'] for r in valid_results]
    
    print("\n" + "="*60)
    print("📊 SUMMARY STATISTICS")
    print("="*60)
    print(f"\nTotal Files Processed: {len(results)}")
    print(f"Successful Predictions: {len(valid_results)}")
    print(f"Failed Predictions: {len(results) - len(valid_results)}")
    
    print(f"\n📈 Valence Statistics:")
    print(f"  Mean:   {sum(valences)/len(valences):+.4f}")
    print(f"  Median: {sorted(valences)[len(valences)//2]:+.4f}")
    print(f"  Min:    {min(valences):+.4f}")
    print(f"  Max:    {max(valences):+.4f}")
    
    print(f"\n📈 Arousal Statistics:")
    print(f"  Mean:   {sum(arousals)/len(arousals):+.4f}")
    print(f"  Median: {sorted(arousals)[len(arousals)//2]:+.4f}")
    print(f"  Min:    {min(arousals):+.4f}")
    print(f"  Max:    {max(arousals):+.4f}")
    
    # Quadrant distribution
    q1 = sum(1 for v, a in zip(valences, arousals) if v > 0 and a > 0)
    q2 = sum(1 for v, a in zip(valences, arousals) if v > 0 and a < 0)
    q3 = sum(1 for v, a in zip(valences, arousals) if v < 0 and a < 0)
    q4 = sum(1 for v, a in zip(valences, arousals) if v < 0 and a > 0)
    
    print(f"\n🎯 Emotion Quadrants:")
    print(f"  Q1 (Happy/Excited):    {q1:3d} ({q1/len(valid_results)*100:.1f}%)")
    print(f"  Q2 (Calm/Peaceful):    {q2:3d} ({q2/len(valid_results)*100:.1f}%)")
    print(f"  Q3 (Sad/Depressed):    {q3:3d} ({q3/len(valid_results)*100:.1f}%)")
    print(f"  Q4 (Angry/Tense):      {q4:3d} ({q4/len(valid_results)*100:.1f}%)")
    print("="*60)


def print_top_predictions(results, n=5):
    """Print top N predictions by various criteria"""
    valid_results = [r for r in results if 'error' not in r]
    
    if not valid_results:
        return
    
    print("\n" + "="*60)
    print(f"🏆 TOP {n} PREDICTIONS")
    print("="*60)
    
    # Most positive
    print(f"\n😊 Most Positive (Highest Valence):")
    top_positive = sorted(valid_results, key=lambda x: x['valence'], reverse=True)[:n]
    for i, r in enumerate(top_positive, 1):
        print(f"  {i}. {r['file'][:40]:40s} Valence: {r['valence']:+.4f}")
    
    # Most negative
    print(f"\n😢 Most Negative (Lowest Valence):")
    top_negative = sorted(valid_results, key=lambda x: x['valence'])[:n]
    for i, r in enumerate(top_negative, 1):
        print(f"  {i}. {r['file'][:40]:40s} Valence: {r['valence']:+.4f}")
    
    # Highest energy
    print(f"\n⚡ Highest Energy (Highest Arousal):")
    top_energy = sorted(valid_results, key=lambda x: x['arousal'], reverse=True)[:n]
    for i, r in enumerate(top_energy, 1):
        print(f"  {i}. {r['file'][:40]:40s} Arousal: {r['arousal']:+.4f}")
    
    # Lowest energy
    print(f"\n💤 Lowest Energy (Lowest Arousal):")
    top_calm = sorted(valid_results, key=lambda x: x['arousal'])[:n]
    for i, r in enumerate(top_calm, 1):
        print(f"  {i}. {r['file'][:40]:40s} Arousal: {r['arousal']:+.4f}")
    
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description='Batch predict emotions from multiple audio files'
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--audio_dir',
        type=str,
        help='Directory containing audio files'
    )
    input_group.add_argument(
        '--audio_files',
        nargs='+',
        help='List of audio file paths'
    )
    
    # Model options
    parser.add_argument(
        '--model',
        type=str,
        default='best_vit',
        choices=['best_vit', 'mobile_vit', 'vit'],
        help='Model to use for prediction'
    )
    parser.add_argument(
        '--model_dir',
        type=str,
        default='../selected/final_best_vit',
        help='Directory containing model checkpoints'
    )
    
    # Processing options
    parser.add_argument(
        '--n_samples',
        type=int,
        default=None,
        help='Number of samples to process (default: all)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='../test/results',
        help='Directory to save results'
    )
    parser.add_argument(
        '--format',
        type=str,
        default='both',
        choices=['csv', 'json', 'both'],
        help='Output format'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cpu', 'cuda'],
        help='Device to run inference on'
    )
    
    args = parser.parse_args()
    
    # Get audio files
    if args.audio_dir:
        print(f"🔍 Scanning directory: {args.audio_dir}")
        audio_files = find_audio_files(args.audio_dir, limit=args.n_samples)
        print(f"   Found {len(audio_files)} audio files")
    else:
        audio_files = args.audio_files
        if args.n_samples:
            audio_files = audio_files[:args.n_samples]
    
    if not audio_files:
        print("❌ No audio files found!")
        sys.exit(1)
    
    print(f"\n📝 Processing {len(audio_files)} audio files...")
    
    # Resolve model path
    model_dir = Path(__file__).parent / args.model_dir
    
    if args.model == 'best_vit':
        model_path = model_dir / 'best_model.pth'
    elif args.model == 'mobile_vit':
        model_path = model_dir / 'mobile_vit_student.pth'
    else:
        model_path = model_dir / 'best_model.pth'
    
    if not model_path.exists():
        print(f"❌ Error: Model not found at {model_path}")
        sys.exit(1)
    
    # Set device
    device = torch.device(args.device) if args.device else None
    
    # Initialize predictor
    print("\n🔄 Loading model...")
    predictor = EmotionPredictor(
        model_path=str(model_path),
        model_type=args.model,
        device=device
    )
    
    # Process files
    print("\n🎵 Processing audio files...")
    print("-" * 60)
    results = predictor.predict_batch(audio_files)
    print("-" * 60)
    
    # Save results
    df = save_results(results, args.output_dir, format=args.format)
    
    # Print statistics
    print_summary_statistics(results)
    print_top_predictions(results, n=5)
    
    print(f"\n✅ Batch prediction complete!")


if __name__ == '__main__':
    main()
