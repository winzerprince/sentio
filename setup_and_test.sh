#!/bin/bash

# Quick Setup and Test Script for AI Music Emotion Analysis
# Optimized for HP EliteBook 840 G3

echo "=============================================="
echo "AI Music Emotion Analysis - Quick Setup"
echo "=============================================="

# Check Python version
python_version=$(python3 --version)
echo "Python version: $python_version"

# Install essential packages first
echo "Installing essential packages..."
pip3 install numpy pandas scikit-learn matplotlib seaborn psutil --quiet

# Install gradient boosting libraries
echo "Installing gradient boosting libraries..."
pip3 install xgboost lightgbm --quiet

# Install audio processing (if needed for generation)
echo "Installing audio processing libraries..."
pip3 install librosa soundfile --quiet

# Test basic imports
echo "Testing imports..."
python3 -c "
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import psutil
print('✓ Core libraries imported successfully')

try:
    import xgboost as xgb
    import lightgbm as lgb
    print('✓ Gradient boosting libraries imported successfully')
except ImportError as e:
    print('⚠ Gradient boosting libraries not available:', e)

try:
    import librosa
    import soundfile as sf
    print('✓ Audio processing libraries imported successfully')
except ImportError as e:
    print('⚠ Audio processing libraries not available:', e)
"

# Check system resources
echo "Checking system resources..."
python3 -c "
import psutil
memory = psutil.virtual_memory()
print(f'Total RAM: {memory.total / (1024**3):.1f}GB')
print(f'Available RAM: {memory.available / (1024**3):.1f}GB')
print(f'CPU cores: {psutil.cpu_count()}')
print(f'CPU frequency: {psutil.cpu_freq().current:.0f}MHz' if psutil.cpu_freq() else 'CPU frequency: Unknown')
"

# Quick data check
echo "Checking dataset availability..."
if [ -d "/mnt/sdb8mount/free-explore/class/ai/datasets/sentio/dataset/DEAM" ]; then
    echo "✓ DEAM dataset directory found"
    
    # Count features files
    feature_count=$(find /mnt/sdb8mount/free-explore/class/ai/datasets/sentio/dataset/DEAM/features -name "*.csv" | wc -l)
    echo "✓ Feature files available: $feature_count"
    
    # Check annotations
    if [ -f "/mnt/sdb8mount/free-explore/class/ai/datasets/sentio/dataset/DEAM/annotations/annotations averaged per song/song_level/static_annotations_averaged_songs_1_2000.csv" ]; then
        echo "✓ Annotation files found"
    else
        echo "⚠ Annotation files not found"
    fi
    
    # Check audio files (optional for generation)
    if [ -d "/mnt/sdb8mount/free-explore/class/ai/datasets/sentio/dataset/DEAM/MEMD_audio" ]; then
        audio_count=$(find /mnt/sdb8mount/free-explore/class/ai/datasets/sentio/dataset/DEAM/MEMD_audio -name "*.mp3" | wc -l 2>/dev/null || echo "0")
        echo "✓ Audio files available: $audio_count (optional for generation)"
    else
        echo "ⅰ Audio directory not found (generation models will be limited)"
    fi
else
    echo "✗ DEAM dataset directory not found"
    echo "Please ensure the dataset is in the correct location"
fi

echo "=============================================="
echo "Setup check completed!"
echo "=============================================="

# Offer to run a quick test
echo ""
echo "Would you like to run a quick analysis test? (y/n)"
read -r response
if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    echo "Running quick analysis test..."
    python3 -c "
import sys
sys.path.append('/mnt/sdb8mount/free-explore/class/ai/datasets/sentio/src')

try:
    from model_testing_framework import DatasetManager, SystemMonitor
    
    print('Testing dataset loading...')
    BASE_PATH = '/mnt/sdb8mount/free-explore/class/ai/datasets/sentio/dataset/DEAM'
    
    # Test with small sample
    dataset_manager = DatasetManager(BASE_PATH, max_samples=50)
    annotations = dataset_manager.load_annotations()
    print(f'✓ Loaded {len(annotations)} annotations')
    
    # Test system monitoring
    monitor = SystemMonitor()
    stats = monitor.get_current_stats()
    print(f'✓ System monitoring working: {stats[\"current_memory_gb\"]:.2f}GB memory')
    
    print('✓ Quick test completed successfully!')
    print('You can now run the full test suite with: python3 run_comprehensive_tests.py')
    
except Exception as e:
    print(f'✗ Quick test failed: {e}')
    print('Check the error and dependencies above')
"
else
    echo "Skipping quick test. You can run it later with:"
    echo "python3 run_comprehensive_tests.py"
fi

echo ""
echo "Setup complete! 🎵"
