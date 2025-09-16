#!/bin/bash

echo "=============================================="
echo "Installing CPU-Only Dependencies"
echo "=============================================="

# Install essential packages first (no NVIDIA dependencies)
echo "Installing core ML packages..."
pip3 install numpy pandas scikit-learn matplotlib seaborn psutil tqdm

echo "Installing gradient boosting (CPU-only)..."
pip3 install xgboost lightgbm

echo "Installing audio processing..."
pip3 install librosa soundfile audioread

echo "Installing music processing..."
pip3 install music21 mido

echo "Installing PyTorch CPU-only version..."
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

echo "Installing TensorFlow CPU-only version..."
pip3 install tensorflow-cpu

echo "Installing Jupyter..."
pip3 install jupyter ipykernel notebook

echo "Installing utilities..."
pip3 install scipy joblib pyyaml python-dotenv h5py hdf5storage

echo "Installing visualization..."
pip3 install plotly

echo "Installing development tools..."
pip3 install pytest black flake8

echo "Installing presentation tools..."
pip3 install python-pptx mistune

echo "Installation complete!"
echo "=============================================="

# Test imports
echo "Testing critical imports..."
python3 -c "
import numpy as np
import pandas as pd
import sklearn
import torch
import librosa
import xgboost
import lightgbm
print('✓ All critical packages imported successfully')
print('PyTorch CUDA available:', torch.cuda.is_available())
print('PyTorch version:', torch.__version__)
"
