#!/bin/bash

# Installation script for test dependencies
# Run this from the project root directory

set -e

echo "=========================================="
echo "📦 Installing Test Dependencies"
echo "=========================================="

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Virtual environment not activated"
    echo "   Please run: source .venv/bin/activate"
    exit 1
fi

echo "✓ Virtual environment: $VIRTUAL_ENV"

# Install core dependencies
echo ""
echo "📦 Installing transformers..."
pip install -q transformers>=4.30.0

echo "📦 Installing Pillow..."
pip install -q pillow>=10.0.0

# Verify installations
echo ""
echo "✅ Verifying installations..."

python -c "import transformers; print(f'  ✓ transformers {transformers.__version__}')" || {
    echo "  ❌ transformers installation failed"
    exit 1
}

python -c "import PIL; print(f'  ✓ PIL {PIL.__version__}')" || {
    echo "  ❌ PIL installation failed"
    exit 1
}

python -c "import torch; print(f'  ✓ torch {torch.__version__}')" || {
    echo "  ❌ torch not installed - please install PyTorch first"
    exit 1
}

python -c "import librosa; print(f'  ✓ librosa {librosa.__version__}')" || {
    echo "  ❌ librosa not installed - please install from requirements.txt"
    exit 1
}

echo ""
echo "=========================================="
echo "✅ All dependencies installed successfully"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  cd test"
echo "  python quick_test.py"
