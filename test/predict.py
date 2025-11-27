#!/usr/bin/env python3
"""
Emotion Prediction from Audio Files

This script takes audio files as input and predicts valence and arousal
using trained emotion recognition models.

Usage:
    python predict.py --audio_file path/to/song.mp3
    python predict.py --audio_file song.mp3 --model best_vit
    python predict.py --audio_file song.mp3 --model mobile_vit
"""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from vit_model import ViTForEmotionRegression, MobileViTStudent
from audio_preprocessor import AudioPreprocessor


class EmotionPredictor:
    """
    Emotion prediction system for audio files.
    Supports multiple model architectures.
    """
    
    def __init__(self, model_path, model_type='vit', device=None):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to model checkpoint (.pth or .pt file)
            model_type: Type of model ('vit', 'mobile_vit', 'best_vit')
            device: Torch device (cuda/cpu), auto-detected if None
        """
        self.model_path = model_path
        self.model_type = model_type.lower()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize preprocessor
        self.preprocessor = AudioPreprocessor()
        
        # Load model
        self.model = self._load_model()
        self.model.eval()
        
        print(f"✅ Model loaded: {model_type}")
        print(f"   Device: {self.device}")
        print(f"   Checkpoint: {model_path}")
    
    def _load_model(self):
        """Load model from checkpoint"""
        try:
            # Initialize model architecture
            if self.model_type in ['vit', 'best_vit']:
                model = ViTForEmotionRegression(
                    num_emotions=2,
                    freeze_backbone=False,
                    dropout=0.1
                )
            elif self.model_type == 'mobile_vit':
                model = MobileViTStudent(
                    num_emotions=2,
                    dropout=0.1
                )
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")
            
            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                else:
                    model.load_state_dict(checkpoint)
            else:
                model.load_state_dict(checkpoint)
            
            model.to(self.device)
            return model
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def predict(self, audio_path):
        """
        Predict valence and arousal for a single audio file.
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            dict: {
                'valence': float (-1 to 1),
                'arousal': float (-1 to 1),
                'valence_normalized': float (0 to 1),
                'arousal_normalized': float (0 to 1)
            }
        """
        # Preprocess audio
        image_tensor, mel_spec = self.preprocessor.preprocess(audio_path)
        
        # Add batch dimension and move to device
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            predictions = self.model(image_tensor)
        
        # Extract values
        valence = predictions[0, 0].item()
        arousal = predictions[0, 1].item()
        
        # Normalize to [0, 1] range for easier interpretation
        valence_norm = (valence + 1) / 2
        arousal_norm = (arousal + 1) / 2
        
        return {
            'valence': valence,
            'arousal': arousal,
            'valence_normalized': valence_norm,
            'arousal_normalized': arousal_norm,
            'file': os.path.basename(audio_path)
        }
    
    def predict_batch(self, audio_paths):
        """
        Predict valence and arousal for multiple audio files.
        
        Args:
            audio_paths: List of audio file paths
        
        Returns:
            list of dicts with predictions
        """
        results = []
        
        for audio_path in audio_paths:
            try:
                result = self.predict(audio_path)
                results.append(result)
                print(f"✓ {result['file']}")
            except Exception as e:
                print(f"✗ {os.path.basename(audio_path)}: {e}")
                results.append({
                    'file': os.path.basename(audio_path),
                    'error': str(e)
                })
        
        return results


def format_emotion_output(result):
    """Format prediction result for display"""
    if 'error' in result:
        return f"\n❌ Error processing {result['file']}: {result['error']}"
    
    output = f"""
{'='*60}
🎵 Audio File: {result['file']}
{'='*60}

📊 Emotion Predictions:

  Valence: {result['valence']:+.4f}  (normalized: {result['valence_normalized']:.4f})
  Arousal: {result['arousal']:+.4f}  (normalized: {result['arousal_normalized']:.4f})

Interpretation:
  - Valence: {interpret_valence(result['valence'])}
  - Arousal: {interpret_arousal(result['arousal'])}
  - Overall: {interpret_emotion(result['valence'], result['arousal'])}

{'='*60}
"""
    return output


def interpret_valence(valence):
    """Interpret valence value"""
    if valence > 0.5:
        return "Very Positive 😊"
    elif valence > 0.1:
        return "Positive 🙂"
    elif valence > -0.1:
        return "Neutral 😐"
    elif valence > -0.5:
        return "Negative 🙁"
    else:
        return "Very Negative 😢"


def interpret_arousal(arousal):
    """Interpret arousal value"""
    if arousal > 0.5:
        return "Very High Energy ⚡"
    elif arousal > 0.1:
        return "High Energy 🔥"
    elif arousal > -0.1:
        return "Moderate Energy 💫"
    elif arousal > -0.5:
        return "Low Energy 😌"
    else:
        return "Very Low Energy 💤"


def interpret_emotion(valence, arousal):
    """Interpret combined valence-arousal into emotion quadrant"""
    if valence > 0 and arousal > 0:
        return "Happy/Excited (High Valence, High Arousal) 🎉"
    elif valence > 0 and arousal < 0:
        return "Calm/Peaceful (High Valence, Low Arousal) 😌"
    elif valence < 0 and arousal > 0:
        return "Angry/Tense (Low Valence, High Arousal) 😠"
    else:
        return "Sad/Depressed (Low Valence, Low Arousal) 😞"


def main():
    parser = argparse.ArgumentParser(
        description='Predict emotion (valence/arousal) from audio files'
    )
    parser.add_argument(
        '--audio_file',
        type=str,
        required=True,
        help='Path to audio file (mp3, wav, etc.)'
    )
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
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cpu', 'cuda'],
        help='Device to run inference on (auto-detected if not specified)'
    )
    
    args = parser.parse_args()
    
    # Resolve model path
    model_dir = Path(__file__).parent / args.model_dir
    
    if args.model == 'best_vit':
        model_path = model_dir / 'best_model.pth'
    elif args.model == 'mobile_vit':
        model_path = model_dir / 'mobile_vit_student.pth'
    else:
        model_path = model_dir / 'best_model.pth'
    
    # Check if model exists
    if not model_path.exists():
        print(f"❌ Error: Model not found at {model_path}")
        print(f"\nAvailable models in {model_dir}:")
        if model_dir.exists():
            for f in model_dir.glob('*.pth'):
                print(f"  - {f.name}")
            for f in model_dir.glob('*.pt'):
                print(f"  - {f.name}")
        else:
            print(f"  Directory does not exist!")
        sys.exit(1)
    
    # Check if audio file exists
    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        print(f"❌ Error: Audio file not found: {audio_path}")
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
    
    # Make prediction
    print(f"\n🎵 Processing audio: {audio_path.name}")
    result = predictor.predict(str(audio_path))
    
    # Display results
    print(format_emotion_output(result))


if __name__ == '__main__':
    main()
