#!/usr/bin/env python3
"""
Download MIT Audio Spectrogram Transformer (AST) Model for Kaggle
================================================================

This script downloads the MIT/ast-finetuned-audioset-10-10-0.4593 model from
Hugging Face Hub and prepares it for upload to Kaggle as a dataset.

Usage:
    python download_mit_ast.py

Output:
    - ./mit-ast-model-for-kaggle/ (model files)
    - ./mit-ast-model-for-kaggle.zip (ready for Kaggle upload)
"""

import os
import sys
import zipfile
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
    print("✅ huggingface_hub is available")
except ImportError:
    print("❌ huggingface_hub not found. Installing...")
    os.system(f"{sys.executable} -m pip install huggingface_hub")
    from huggingface_hub import snapshot_download

# Configuration
MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"
LOCAL_DIR = "./mit-ast-model-for-kaggle"
ZIP_FILE = "./mit-ast-model-for-kaggle.zip"

def format_size(size_bytes):
    """Format file size in human readable format."""
    if size_bytes == 0:
        return "0B"
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024.0 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    return f"{size_bytes:.1f}{size_names[i]}"

def get_directory_size(directory):
    """Calculate total size of directory."""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.exists(filepath):
                total_size += os.path.getsize(filepath)
    return total_size

def download_model():
    """Download MIT AST model from Hugging Face Hub."""
    print("=" * 60)
    print("🤖 MIT AST Model Downloader for Kaggle")
    print("=" * 60)
    print(f"Model: {MODEL_NAME}")
    print(f"Target Directory: {LOCAL_DIR}")
    print(f"Output ZIP: {ZIP_FILE}")
    print()
    
    try:
        # Remove existing directory if it exists
        if os.path.exists(LOCAL_DIR):
            print(f"🗑️  Removing existing directory: {LOCAL_DIR}")
            import shutil
            shutil.rmtree(LOCAL_DIR)
        
        print(f"📥 Downloading MIT AST model...")
        print(f"   This may take several minutes depending on your internet speed.")
        print()
        
        # Download the model
        snapshot_download(
            repo_id=MODEL_NAME,
            local_dir=LOCAL_DIR,
            resume_download=True,
            local_files_only=False
        )
        
        print(f"✅ Successfully downloaded MIT AST model!")
        
        # Check downloaded files
        if os.path.exists(LOCAL_DIR):
            total_size = get_directory_size(LOCAL_DIR)
            print(f"📊 Downloaded size: {format_size(total_size)}")
            
            # List key files
            key_files = []
            for root, dirs, files in os.walk(LOCAL_DIR):
                for file in files:
                    if file.endswith(('.json', '.bin', '.safetensors', '.txt')):
                        file_path = os.path.join(root, file)
                        file_size = os.path.getsize(file_path)
                        rel_path = os.path.relpath(file_path, LOCAL_DIR)
                        key_files.append((rel_path, file_size))
            
            print(f"📁 Key model files:")
            for file_name, file_size in sorted(key_files):
                print(f"   - {file_name} ({format_size(file_size)})")
        
        return True
        
    except Exception as e:
        print(f"❌ Download failed: {str(e)}")
        print()
        print("🔧 Troubleshooting:")
        print("   1. Check your internet connection")
        print("   2. Verify Hugging Face Hub access")
        print("   3. Try running: pip install --upgrade huggingface_hub")
        print("   4. Check if the model name is correct")
        return False

def create_zip():
    """Create ZIP file for Kaggle upload."""
    if not os.path.exists(LOCAL_DIR):
        print(f"❌ Model directory not found: {LOCAL_DIR}")
        return False
    
    try:
        print(f"📦 Creating ZIP file: {ZIP_FILE}")
        
        # Remove existing ZIP if it exists
        if os.path.exists(ZIP_FILE):
            os.remove(ZIP_FILE)
        
        # Create ZIP file
        with zipfile.ZipFile(ZIP_FILE, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
            for root, dirs, files in os.walk(LOCAL_DIR):
                for file in files:
                    file_path = os.path.join(root, file)
                    arc_name = os.path.relpath(file_path, '.')
                    zipf.write(file_path, arc_name)
                    print(f"   Adding: {arc_name}")
        
        # Check ZIP file size
        zip_size = os.path.getsize(ZIP_FILE)
        print(f"✅ ZIP created successfully!")
        print(f"📊 ZIP file size: {format_size(zip_size)}")
        
        return True
        
    except Exception as e:
        print(f"❌ ZIP creation failed: {str(e)}")
        return False

def print_kaggle_instructions():
    """Print instructions for Kaggle upload."""
    print()
    print("=" * 60)
    print("📋 KAGGLE UPLOAD INSTRUCTIONS")
    print("=" * 60)
    print("1. 🌐 Go to https://kaggle.com and login")
    print("2. ➕ Click 'Create' → 'New Dataset'")
    print(f"3. 📁 Upload the file: {ZIP_FILE}")
    print("4. 📝 Fill in dataset details:")
    print("   - Title: 'MIT Audio Spectrogram Transformer AST'")
    print("   - Slug: 'mit-ast-model-kaggle'")
    print("   - Description: 'Pre-trained MIT AST model for audio classification'")
    print("5. 🚀 Click 'Create Dataset'")
    print()
    print("6. 📓 In your notebook, add this dataset as input")
    print("7. 🔧 Update your notebook configuration:")
    print("   AST_MODEL_NAME = '/kaggle/input/mit-ast-model-kaggle/mit-ast-model-for-kaggle'")
    print()
    print("✨ Your notebook will then load the model locally!")
    print("=" * 60)

def main():
    """Main function."""
    print("🚀 Starting MIT AST model download...")
    
    # Download model
    if download_model():
        print()
        
        # Create ZIP file
        if create_zip():
            print_kaggle_instructions()
            print("🎉 Process completed successfully!")
            return True
    
    print("💥 Process failed. Please check the errors above.")
    return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
