#!/bin/bash

# Download ViT Model for Kaggle Upload
# ====================================
# This script downloads the Google ViT model to your local machine
# so you can upload it as a Kaggle dataset.

echo "🚀 Downloading Google ViT Model for Kaggle Upload"
echo "============================================================"

# Check if huggingface-cli is installed
if ! command -v huggingface-cli &> /dev/null; then
    echo "❌ huggingface-cli not found!"
    echo "📦 Installing huggingface_hub..."
    pip install -U huggingface_hub
fi

# Model details
MODEL_NAME="google/vit-base-patch16-224-in21k"
OUTPUT_DIR="vit-model-for-kaggle"

echo "📦 Model: $MODEL_NAME"
echo "📁 Output directory: $OUTPUT_DIR"

# Remove existing directory if it exists
if [ -d "$OUTPUT_DIR" ]; then
    echo "🗑️ Removing existing directory: $OUTPUT_DIR"
    rm -rf "$OUTPUT_DIR"
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo ""
echo "⏬ Starting download..."
echo "This may take 5-15 minutes depending on your connection"
echo ""

# Download the model
huggingface-cli download \
    --resume-download \
    --local-dir "$OUTPUT_DIR" \
    "$MODEL_NAME"

# Check if download was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Download completed!"
    echo "📁 Model saved to: $(pwd)/$OUTPUT_DIR"
    
    echo ""
    echo "📋 Downloaded files:"
    ls -lh "$OUTPUT_DIR"
    
    echo ""
    echo "📊 Total size:"
    du -sh "$OUTPUT_DIR"
    
    echo ""
    echo "============================================================"
    echo "🎯 NEXT STEPS:"
    echo "============================================================"
    echo "1. Create a ZIP file:"
    echo "   zip -r vit-model-kaggle.zip $OUTPUT_DIR"
    echo ""
    echo "2. Go to kaggle.com and create a new dataset:"
    echo "   - Click 'New Dataset'"
    echo "   - Upload vit-model-kaggle.zip"
    echo "   - Title: 'ViT Base Patch16 224 ImageNet21k'"
    echo "   - Make it public"
    echo ""
    echo "3. In your Kaggle notebook:"
    echo "   - Add your dataset via '+ Add Data'"
    echo "   - Update model path: /kaggle/input/your-dataset-name/"
    echo ""
    echo "✅ Model ready for Kaggle upload!"
    
    # Automatically create ZIP if zip command is available
    if command -v zip &> /dev/null; then
        echo ""
        echo "📦 Creating ZIP file..."
        zip -r vit-model-kaggle.zip "$OUTPUT_DIR"
        echo "✅ ZIP file created: vit-model-kaggle.zip"
    fi
    
else
    echo ""
    echo "❌ Download failed!"
    echo ""
    echo "💡 Troubleshooting:"
    echo "1. Check your internet connection"
    echo "2. Install huggingface_hub: pip install huggingface_hub"
    echo "3. For authentication: huggingface-cli login"
    echo "4. Try with VPN if geo-blocked"
    exit 1
fi
