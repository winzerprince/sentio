#!/bin/bash
# Download MIT Audio Spectrogram Transformer (AST) Model for Kaggle
# ================================================================
# 
# This script downloads the MIT/ast-finetuned-audioset-10-10-0.4593 model from
# Hugging Face Hub and prepares it for upload to Kaggle as a dataset.
#
# Usage: chmod +x download_mit_ast.sh && ./download_mit_ast.sh

set -e  # Exit on any error

# Configuration
MODEL_NAME="MIT/ast-finetuned-audioset-10-10-0.4593"
LOCAL_DIR="./mit-ast-model-for-kaggle"
ZIP_FILE="./mit-ast-model-for-kaggle.zip"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}🤖 MIT AST Model Downloader for Kaggle${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "Model: ${MODEL_NAME}"
echo -e "Target Directory: ${LOCAL_DIR}"
echo -e "Output ZIP: ${ZIP_FILE}"
echo ""

# Check if huggingface-cli is installed
if ! command -v huggingface-cli &> /dev/null; then
    echo -e "${YELLOW}⚠️  huggingface-cli not found. Installing...${NC}"
    pip install --upgrade huggingface_hub
    
    if ! command -v huggingface-cli &> /dev/null; then
        echo -e "${RED}❌ Failed to install huggingface-cli${NC}"
        echo -e "${YELLOW}💡 Try: pip install huggingface_hub${NC}"
        exit 1
    fi
    echo -e "${GREEN}✅ huggingface-cli installed successfully!${NC}"
fi

# Remove existing directory if it exists
if [ -d "$LOCAL_DIR" ]; then
    echo -e "${YELLOW}🗑️  Removing existing directory: ${LOCAL_DIR}${NC}"
    rm -rf "$LOCAL_DIR"
fi

# Remove existing ZIP if it exists
if [ -f "$ZIP_FILE" ]; then
    echo -e "${YELLOW}🗑️  Removing existing ZIP: ${ZIP_FILE}${NC}"
    rm -f "$ZIP_FILE"
fi

echo -e "${BLUE}📥 Downloading MIT AST model...${NC}"
echo -e "${YELLOW}   This may take several minutes depending on your internet speed.${NC}"
echo ""

# Download the model with resume capability
if huggingface-cli download "$MODEL_NAME" --local-dir "$LOCAL_DIR" --resume-download; then
    echo -e "${GREEN}✅ Successfully downloaded MIT AST model!${NC}"
else
    echo -e "${RED}❌ Download failed!${NC}"
    echo ""
    echo -e "${BLUE}🔧 Troubleshooting:${NC}"
    echo -e "   1. Check your internet connection"
    echo -e "   2. Verify Hugging Face Hub access"
    echo -e "   3. Try running: pip install --upgrade huggingface_hub"
    echo -e "   4. Check if the model name is correct"
    exit 1
fi

# Check downloaded files and calculate size
if [ -d "$LOCAL_DIR" ]; then
    TOTAL_SIZE=$(du -sh "$LOCAL_DIR" | cut -f1)
    FILE_COUNT=$(find "$LOCAL_DIR" -type f | wc -l)
    echo -e "${BLUE}📊 Downloaded: ${TOTAL_SIZE} (${FILE_COUNT} files)${NC}"
    
    echo -e "${BLUE}📁 Key model files:${NC}"
    find "$LOCAL_DIR" -name "*.json" -o -name "*.bin" -o -name "*.safetensors" -o -name "*.txt" | while read file; do
        if [ -f "$file" ]; then
            SIZE=$(du -sh "$file" | cut -f1)
            BASENAME=$(basename "$file")
            echo -e "   - ${BASENAME} (${SIZE})"
        fi
    done
fi

echo ""
echo -e "${BLUE}📦 Creating ZIP file: ${ZIP_FILE}${NC}"

# Create ZIP file
if zip -r "$ZIP_FILE" "$LOCAL_DIR" > /dev/null 2>&1; then
    ZIP_SIZE=$(du -sh "$ZIP_FILE" | cut -f1)
    echo -e "${GREEN}✅ ZIP created successfully!${NC}"
    echo -e "${BLUE}📊 ZIP file size: ${ZIP_SIZE}${NC}"
else
    echo -e "${RED}❌ ZIP creation failed!${NC}"
    echo -e "${YELLOW}💡 Make sure 'zip' is installed: sudo apt-get install zip${NC}"
    exit 1
fi

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}📋 KAGGLE UPLOAD INSTRUCTIONS${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "1. 🌐 Go to https://kaggle.com and login"
echo -e "2. ➕ Click 'Create' → 'New Dataset'"
echo -e "3. 📁 Upload the file: ${ZIP_FILE}"
echo -e "4. 📝 Fill in dataset details:"
echo -e "   - Title: 'MIT Audio Spectrogram Transformer AST'"
echo -e "   - Slug: 'mit-ast-model-kaggle'"
echo -e "   - Description: 'Pre-trained MIT AST model for audio classification'"
echo -e "5. 🚀 Click 'Create Dataset'"
echo ""
echo -e "6. 📓 In your notebook, add this dataset as input"
echo -e "7. 🔧 Update your notebook configuration:"
echo -e "   AST_MODEL_NAME = '/kaggle/input/mit-ast-model-kaggle/mit-ast-model-for-kaggle'"
echo ""
echo -e "${GREEN}✨ Your notebook will then load the model locally!${NC}"
echo -e "${BLUE}========================================${NC}"

# Clean up instructions
echo ""
echo -e "${YELLOW}🧹 Cleanup (optional):${NC}"
echo -e "   - Keep ZIP file for Kaggle upload: ${ZIP_FILE}"
echo -e "   - Remove extracted folder: rm -rf ${LOCAL_DIR}"
echo ""
echo -e "${GREEN}🎉 Process completed successfully!${NC}"
