"""
Audio Preprocessing for Model Inference

This module handles loading audio files and converting them to spectrograms
for input to the emotion prediction models.
"""

import librosa
import numpy as np
import torch
from PIL import Image


class AudioPreprocessor:
    """
    Preprocessor for converting audio files to model-ready spectrograms.
    """
    
    def __init__(self, 
                 sample_rate=22050,
                 duration=30,
                 n_mels=128,
                 hop_length=512,
                 n_fft=2048,
                 fmin=20,
                 fmax=8000,
                 image_size=224):
        """
        Initialize audio preprocessor.
        
        Args:
            sample_rate: Audio sampling rate (Hz)
            duration: Audio clip duration (seconds)
            n_mels: Number of mel-frequency bins
            hop_length: Hop length for STFT
            n_fft: FFT window size
            fmin: Minimum frequency
            fmax: Maximum frequency
            image_size: Target image size for model input (224 for ViT)
        """
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.fmin = fmin
        self.fmax = fmax
        self.image_size = image_size
        
        # ImageNet normalization (used by ViT)
        self.imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    def load_audio(self, audio_path):
        """
        Load audio file.
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            audio: Audio waveform
            sr: Sample rate
        """
        try:
            audio, sr = librosa.load(
                audio_path,
                sr=self.sample_rate,
                duration=self.duration,
                mono=True
            )
            
            # Pad or truncate to exact duration
            target_length = self.sample_rate * self.duration
            if len(audio) < target_length:
                audio = np.pad(audio, (0, target_length - len(audio)))
            else:
                audio = audio[:target_length]
            
            return audio, sr
            
        except Exception as e:
            raise RuntimeError(f"Failed to load audio from {audio_path}: {e}")
    
    def audio_to_melspectrogram(self, audio):
        """
        Convert audio waveform to mel spectrogram.
        
        Args:
            audio: Audio waveform
        
        Returns:
            mel_spec: Mel spectrogram in dB scale
        """
        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            fmin=self.fmin,
            fmax=self.fmax
        )
        
        # Convert to dB scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        return mel_spec_db
    
    def spectrogram_to_image(self, mel_spec):
        """
        Convert mel spectrogram to RGB image tensor for ViT input.
        
        Args:
            mel_spec: Mel spectrogram (n_mels, time_steps)
        
        Returns:
            image_tensor: Tensor of shape (3, 224, 224) normalized for ViT
        """
        # Normalize to [0, 1]
        spec_min = mel_spec.min()
        spec_max = mel_spec.max()
        spec_norm = (mel_spec - spec_min) / (spec_max - spec_min + 1e-8)
        
        # Resize to 224x224 using PIL
        spec_pil = Image.fromarray((spec_norm * 255).astype(np.uint8))
        spec_resized = spec_pil.resize(
            (self.image_size, self.image_size),
            Image.Resampling.BILINEAR
        )
        
        # Convert back to numpy and normalize
        spec_array = np.array(spec_resized).astype(np.float32) / 255.0
        
        # Convert grayscale to RGB by replicating channels
        spec_rgb = np.stack([spec_array, spec_array, spec_array], axis=0)
        
        # Convert to torch tensor
        image_tensor = torch.from_numpy(spec_rgb).float()
        
        # Apply ImageNet normalization
        image_tensor = (image_tensor - self.imagenet_mean) / self.imagenet_std
        
        return image_tensor
    
    def preprocess(self, audio_path):
        """
        Complete preprocessing pipeline: audio file -> model-ready tensor.
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            image_tensor: Tensor of shape (3, 224, 224) ready for model input
            mel_spec: Raw mel spectrogram (for visualization)
        """
        # Load audio
        audio, _ = self.load_audio(audio_path)
        
        # Convert to mel spectrogram
        mel_spec = self.audio_to_melspectrogram(audio)
        
        # Convert to image tensor
        image_tensor = self.spectrogram_to_image(mel_spec)
        
        return image_tensor, mel_spec
    
    def preprocess_batch(self, audio_paths):
        """
        Preprocess multiple audio files.
        
        Args:
            audio_paths: List of audio file paths
        
        Returns:
            batch_tensor: Tensor of shape (batch_size, 3, 224, 224)
            mel_specs: List of mel spectrograms
        """
        tensors = []
        mel_specs = []
        
        for audio_path in audio_paths:
            tensor, mel_spec = self.preprocess(audio_path)
            tensors.append(tensor)
            mel_specs.append(mel_spec)
        
        # Stack into batch
        batch_tensor = torch.stack(tensors, dim=0)
        
        return batch_tensor, mel_specs
