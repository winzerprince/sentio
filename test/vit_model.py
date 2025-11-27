"""
Vision Transformer (ViT) Model Definition for Emotion Regression

This file defines the ViT model architecture used for valence-arousal prediction.
"""

import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig


class ViTForEmotionRegression(nn.Module):
    """
    Vision Transformer for emotion regression (valence and arousal prediction).
    
    Architecture:
    - Pre-trained ViT backbone (google/vit-base-patch16-224-in21k)
    - Custom regression head for 2D emotion prediction
    - Dropout for regularization
    """
    
    def __init__(self, model_name='google/vit-base-patch16-224-in21k', 
                 num_emotions=2, freeze_backbone=False, dropout=0.1):
        super().__init__()
        
        # Load pre-trained ViT model
        try:
            self.vit = ViTModel.from_pretrained(model_name)
            print(f"✅ Loaded pre-trained ViT from {model_name}")
        except Exception as e:
            print(f"⚠️ Could not load pre-trained model: {e}")
            print("   Initializing with random weights...")
            config = ViTConfig()
            self.vit = ViTModel(config)
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.vit.parameters():
                param.requires_grad = False
            print(f"❄️ Frozen ViT backbone")
        
        # Get hidden size from ViT config
        hidden_size = self.vit.config.hidden_size
        
        # Regression head for emotion prediction (named 'head' to match saved checkpoint)
        # Architecture: 768 -> 512 -> 128 -> 2
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),  # [0] weight: [768], bias: [768]
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 512),  # [2] weight: [512, 768], bias: [512]
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),  # [5] weight: [128, 512], bias: [128]
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_emotions),  # [8] weight: [2, 128], bias: [2]
            nn.Tanh()  # Output in range [-1, 1]
        )
    
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            pixel_values: Input images tensor of shape (batch_size, 3, 224, 224)
        
        Returns:
            Emotion predictions tensor of shape (batch_size, 2) [valence, arousal]
        """
        # Get ViT outputs
        outputs = self.vit(pixel_values)
        cls_output = outputs.last_hidden_state[:, 0]
        
        # Predict emotions
        emotion_predictions = self.head(cls_output)
        return emotion_predictions


class MobileViTStudent(nn.Module):
    """
    Lightweight MobileViT student model for emotion regression.
    Used in distilled version for faster inference.
    """
    
    def __init__(self, num_emotions=2, dropout=0.1):
        super().__init__()
        
        # Lightweight CNN backbone
        self.conv_stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        
        # Mobile inverted bottleneck blocks
        self.blocks = nn.Sequential(
            self._make_mb_block(32, 64, stride=2),
            self._make_mb_block(64, 128, stride=2),
            self._make_mb_block(128, 256, stride=2),
        )
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Regression head (named 'head' to match saved checkpoint)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_emotions),
            nn.Tanh()
        )
    
    def _make_mb_block(self, in_channels, out_channels, stride=1):
        """Create Mobile Inverted Bottleneck block"""
        return nn.Sequential(
            # Depthwise
            nn.Conv2d(in_channels, in_channels, kernel_size=3, 
                     stride=stride, padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            # Pointwise
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        """Forward pass"""
        x = self.conv_stem(x)
        x = self.blocks(x)
        x = self.global_pool(x)
        emotions = self.head(x)
        return emotions
