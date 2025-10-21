# Local ViT Model Setup for Kaggle

This document explains how to use the local model download approach to avoid Hugging Face download issues in Kaggle notebooks.

## 🎯 Problem Statement

The notebook `vit_with_gans_emotion_prediction.ipynb` requires the `google/vit-base-patch16-224-in21k` model (346MB), but Kaggle environments often experience:
- 500 Internal Server Errors from Hugging Face
- Network timeouts on large file downloads
- CAS service failures with exponential retry loops

## 🔧 Solution: Local Download + Kaggle Dataset

### Step 1: Download Model Locally

You have two options for downloading the model to your local machine:

#### Option A: Python Script
```bash
python download_vit_model.py
```

#### Option B: Bash Script
```bash
./download_vit_model.sh
```

Both scripts will:
- Download the complete ViT model to `./vit-model-for-kaggle/`
- Create a ZIP file `vit-model-for-kaggle.zip`
- Provide Kaggle upload instructions

### Step 2: Upload to Kaggle as Dataset

1. **Login to Kaggle**: Go to [kaggle.com](https://kaggle.com)

2. **Create New Dataset**: 
   - Click "Create" → "New Dataset"
   - Upload the `vit-model-for-kaggle.zip` file
   - Set title: "Vision Transformer ViT Base Patch16 224 In21k"
   - Set slug: "vit-model-kaggle" 
   - Make it public or private as needed

3. **Get Dataset Path**: After upload, note the dataset path:
   ```
   /kaggle/input/vit-model-kaggle/vit-model-for-kaggle/
   ```

### Step 3: Update Notebook Configuration

In the notebook configuration cell, update the `VIT_MODEL_NAME`:

```python
# ========================
# VIT MODEL CONFIGURATION
# ========================
# OPTION 1: Use pre-downloaded model (recommended)
VIT_MODEL_NAME = '/kaggle/input/vit-model-kaggle/vit-model-for-kaggle'  # ✅ Your dataset path

# OPTION 2: Fallback to online download (may fail)
# VIT_MODEL_NAME = 'google/vit-base-patch16-224-in21k'
```

### Step 4: Add Dataset to Notebook

1. **Edit Notebook**: In Kaggle, click "Edit" on your notebook
2. **Add Input**: Click "Add data" → "Datasets" 
3. **Search**: Find your uploaded dataset
4. **Add**: Click "Add" to include it as input

## 🔄 How the Code Works

The updated `ViTForEmotionRegression` class now includes:

### Local Model Detection
```python
def _load_vit_model(self, model_name):
    if os.path.exists(model_name):
        return self._load_local_model(model_name)  # 🗂️ Local path
    else:
        return self._load_online_model(model_name)  # 🌐 Online download
```

### Robust Local Loading
```python
def _load_local_model(self, model_path):
    # ✅ Verify path exists
    # ✅ Check required files (config.json, etc.)
    # ✅ Load with local_files_only=True
    # ✅ Fallback to online if local fails
```

### Graceful Fallback
If local loading fails, the code automatically falls back to online download with retry logic.

## 📁 Expected File Structure

After successful setup, your Kaggle notebook will have:

```
/kaggle/input/
├── deam-mediaeval-dataset-emotional-analysis-in-music/  # DEAM dataset
└── vit-model-kaggle/                                   # Your ViT model
    └── vit-model-for-kaggle/
        ├── config.json
        ├── model.safetensors
        ├── preprocessor_config.json
        └── ...
```

## 🚨 Troubleshooting

### "Model path does not exist"
- ✅ Check dataset is added to notebook inputs
- ✅ Verify the exact path in Kaggle (case-sensitive)
- ✅ Ensure ZIP was extracted properly

### "Missing model files"
- ✅ Re-download using the provided scripts
- ✅ Check internet connection during download
- ✅ Verify ZIP file is not corrupted

### Local download fails
- ✅ Install required packages: `pip install huggingface_hub`
- ✅ Check internet connection and firewall
- ✅ Try the bash script alternative

## 💡 Benefits of This Approach

1. **Reliability**: No more Hugging Face 500 errors in Kaggle
2. **Speed**: Faster loading from local dataset vs. network download
3. **Offline**: Works without internet in Kaggle environment
4. **Reusable**: Dataset can be shared across multiple notebooks
5. **Version Control**: Pin exact model version for reproducibility

## ⚡ Quick Start Summary

```bash
# 1. Download locally
python download_vit_model.py

# 2. Upload vit-model-for-kaggle.zip to Kaggle as dataset

# 3. Update notebook configuration
VIT_MODEL_NAME = '/kaggle/input/your-dataset-name/vit-model-for-kaggle'

# 4. Add dataset to notebook inputs

# 5. Run notebook - it will load from local path! 🎉
```

This approach eliminates the download reliability issues and provides a smooth experience for running the ViT emotion prediction notebook.
