#!/bin/bash

# Test Suite Setup and Execution Script
# This script sets up the environment and runs tests

set -e  # Exit on error

echo "=========================================="
echo "🎵 Emotion Prediction Test Setup"
echo "=========================================="

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Activate virtual environment
if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    echo "✓ Activating virtual environment..."
    source "$PROJECT_ROOT/.venv/bin/activate"
else
    echo "❌ Virtual environment not found at $PROJECT_ROOT/.venv"
    echo "   Please create it first: python -m venv .venv"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $PYTHON_VERSION"

# Install/update dependencies
echo ""
echo "📦 Checking dependencies..."
pip install -q --upgrade pip

# Check if torch is installed
if ! python -c "import torch" 2>/dev/null; then
    echo "⚠️  PyTorch not found. Installing CPU version..."
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Check if transformers is installed
if ! python -c "import transformers" 2>/dev/null; then
    echo "⚠️  Transformers not found. Installing..."
    pip install transformers>=4.30.0
fi

# Check if librosa is installed
if ! python -c "import librosa" 2>/dev/null; then
    echo "⚠️  Librosa not found. Installing..."
    pip install librosa>=0.10.0
fi

# Install all requirements
echo "📦 Installing all requirements..."
pip install -q -r "$PROJECT_ROOT/requirements.txt"

echo ""
echo "✅ Dependencies installed successfully"

# Check model files
echo ""
echo "🔍 Checking model files..."
MODEL_DIR="$PROJECT_ROOT/selected/final_best_vit"

if [ ! -d "$MODEL_DIR" ]; then
    echo "❌ Model directory not found: $MODEL_DIR"
    exit 1
fi

if [ ! -f "$MODEL_DIR/best_model.pth" ]; then
    echo "❌ Main model not found: $MODEL_DIR/best_model.pth"
    echo "   Available files:"
    ls -lh "$MODEL_DIR"
    exit 1
fi

echo "✓ Found model: best_model.pth"

# Check audio files
echo ""
echo "🔍 Checking audio files..."
AUDIO_DIR="$PROJECT_ROOT/dataset/DEAM/MEMD_audio"

if [ ! -d "$AUDIO_DIR" ]; then
    echo "❌ Audio directory not found: $AUDIO_DIR"
    exit 1
fi

AUDIO_COUNT=$(ls -1 "$AUDIO_DIR"/*.mp3 2>/dev/null | wc -l)
echo "✓ Found $AUDIO_COUNT audio files"

# Create results directory
mkdir -p "$SCRIPT_DIR/results"
echo "✓ Results directory ready: $SCRIPT_DIR/results"

# Run tests
echo ""
echo "=========================================="
echo "🧪 Running Quick Test"
echo "=========================================="
python "$SCRIPT_DIR/quick_test.py"

echo ""
echo "=========================================="
echo "✅ Test Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Single file test:  python test/predict.py --audio_file dataset/DEAM/MEMD_audio/10.mp3"
echo "  2. Batch test:        python test/batch_predict.py --audio_dir dataset/DEAM/MEMD_audio --n_samples 10"
echo "  3. View results:      cat test/results/*.csv"
echo ""
