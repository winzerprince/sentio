#!/usr/bin/env python3
"""
Download Google ViT Model for Kaggle Upload
===========================================

This script downloads the ViT model locally so you can upload it as a Kaggle dataset.
Run this on your local machine where you have stable internet.

Usage:
    python download_vit_model.py

Requirements:
    pip install huggingface_hub transformers torch
"""

import os
from pathlib import Path
from huggingface_hub import snapshot_download
import shutil

def download_vit_model():
    """Download ViT model and prepare for Kaggle upload"""
    
    print("🚀 Downloading Google ViT Model for Kaggle Upload")
    print("=" * 60)
    
    # Model details
    model_name = "google/vit-base-patch16-224-in21k"
    output_dir = "vit-model-for-kaggle"
    
    print(f"📦 Model: {model_name}")
    print(f"📁 Output directory: {output_dir}")
    
    # Create output directory
    if os.path.exists(output_dir):
        print(f"🗑️ Removing existing directory: {output_dir}")
        shutil.rmtree(output_dir)
    
    os.makedirs(output_dir)
    
    try:
        print("\n⏬ Starting download...")
        print("This may take 5-15 minutes depending on your connection")
        
        # Download model files
        model_path = snapshot_download(
            repo_id=model_name,
            cache_dir="./cache",
            local_dir=output_dir,
            resume_download=True,
            ignore_patterns=[
                "*.git*",
                "*.md",
                "*.txt", 
                "flax_*",  # Skip Flax files (we only need PyTorch)
                "tf_*"     # Skip TensorFlow files
            ]
        )
        
        print(f"\n✅ Download completed!")
        print(f"📁 Model saved to: {os.path.abspath(output_dir)}")
        
        # List downloaded files
        print("\n📋 Downloaded files:")
        for file in sorted(os.listdir(output_dir)):
            file_path = os.path.join(output_dir, file)
            if os.path.isfile(file_path):
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                print(f"   {file:<30} ({size_mb:.1f} MB)")
        
        # Calculate total size
        total_size = sum(
            os.path.getsize(os.path.join(output_dir, f))
            for f in os.listdir(output_dir)
            if os.path.isfile(os.path.join(output_dir, f))
        )
        total_mb = total_size / (1024 * 1024)
        
        print(f"\n📊 Total size: {total_mb:.1f} MB")
        
        print("\n" + "=" * 60)
        print("🎯 NEXT STEPS:")
        print("=" * 60)
        print("1. Create a ZIP file of the model folder:")
        print(f"   zip -r vit-model-kaggle.zip {output_dir}")
        print("")
        print("2. Go to kaggle.com and create a new dataset:")
        print("   - Click 'New Dataset'")
        print("   - Upload vit-model-kaggle.zip")
        print("   - Title: 'ViT Base Patch16 224 ImageNet21k'")
        print("   - Make it public")
        print("")
        print("3. In your Kaggle notebook:")
        print("   - Add your dataset via '+ Add Data'")
        print("   - Update model path in code")
        print("")
        print("✅ Model ready for Kaggle upload!")
        
    except Exception as e:
        print(f"\n❌ Error downloading model: {e}")
        print("\n💡 Troubleshooting:")
        print("1. Check your internet connection")
        print("2. Install required packages: pip install huggingface_hub transformers")
        print("3. For authentication issues: huggingface-cli login")
        print("4. Try using a VPN if geo-blocked")
        return False
    
    return True

if __name__ == "__main__":
    success = download_vit_model()
    if not success:
        exit(1)
