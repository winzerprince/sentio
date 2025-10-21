# MIT AST Local Model Setup for Kaggle

This document explains how to use the local model download approach for the MIT Audio Spectrogram Transformer notebook to avoid Hugging Face download issues in Kaggle.

## 🎯 Problem Statement

The notebook `mit_ast_with_gans_emotion_prediction.ipynb` requires the `MIT/ast-finetuned-audioset-10-10-0.4593` model, but Kaggle environments often experience:
- Network timeouts on model downloads
- Hugging Face Hub connectivity issues  
- Server-side errors during model loading
- Memory constraints during download

## 🔧 Solution: Local Download + Kaggle Dataset

### Step 1: Download MIT AST Model Locally

You have two options for downloading the model to your local machine:

#### Option A: Python Script
```bash
python download_mit_ast.py
```

#### Option B: Bash Script  
```bash
chmod +x download_mit_ast.sh
./download_mit_ast.sh
```

Both scripts will:
- Download the complete MIT AST model to `./mit-ast-model-for-kaggle/`
- Create a ZIP file `mit-ast-model-for-kaggle.zip`
- Provide detailed Kaggle upload instructions

### Step 2: Upload to Kaggle as Dataset

1. **Login to Kaggle**: Go to [kaggle.com](https://kaggle.com)

2. **Create New Dataset**: 
   - Click "Create" → "New Dataset"
   - Upload the `mit-ast-model-for-kaggle.zip` file
   - Set title: "MIT Audio Spectrogram Transformer AST"
   - Set slug: "mit-ast-model-kaggle"
   - Make it public or private as needed

3. **Get Dataset Path**: After upload, note the dataset path:
   ```
   /kaggle/input/mit-ast-model-kaggle/mit-ast-model-for-kaggle/
   ```

### Step 3: Update Notebook Configuration

In the notebook configuration cell, update the `AST_MODEL_NAME`:

```python
# ========================
# AST MODEL CONFIGURATION
# ========================
# OPTION 1: Use pre-downloaded model (recommended)
AST_MODEL_NAME = '/kaggle/input/mit-ast-model-kaggle/mit-ast-model-for-kaggle'  # ✅ Your dataset path

# OPTION 2: Fallback to online download (may fail)
# AST_MODEL_NAME = 'MIT/ast-finetuned-audioset-10-10-0.4593'
```

### Step 4: Add Dataset to Notebook

1. **Edit Notebook**: In Kaggle, click "Edit" on your notebook
2. **Add Input**: Click "Add data" → "Datasets"
3. **Search**: Find your uploaded MIT AST dataset
4. **Add**: Click "Add" to include it as input

## 🔄 How the Code Works

The updated `ASTForEmotionRegression` class includes:

### Local Model Detection
```python
def _load_ast_model(self, model_name):
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
    # ✅ Load both model and feature extractor locally
    # ✅ Fallback to online if local fails
```

### Graceful Fallback
If local loading fails, the code automatically falls back to online download with retry logic.

## 📁 Expected File Structure

After successful setup, your Kaggle notebook will have:

```
/kaggle/input/
├── deam-mediaeval-dataset-emotional-analysis-in-music/  # DEAM dataset
└── mit-ast-model-kaggle/                               # Your AST model
    └── mit-ast-model-for-kaggle/
        ├── config.json
        ├── pytorch_model.bin
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
- ✅ Verify model is complete (check file sizes)

### Local download fails
- ✅ Install required packages: `pip install huggingface_hub`
- ✅ Check internet connection and firewall
- ✅ Try the bash script alternative
- ✅ Use `huggingface-cli login` if authentication needed

### AST feature extractor issues
- ✅ Ensure both model and feature extractor are downloaded
- ✅ Check `preprocessor_config.json` exists
- ✅ Verify audio preprocessing parameters match AST requirements

## 🎵 MIT AST Specific Considerations

### Audio Requirements
- **Sample Rate**: 16kHz (different from ViT's 22kHz)
- **Duration**: 10 seconds (optimized for AST)
- **Input Format**: Raw audio → AST feature extractor → model

### Model Architecture
- **Input**: Audio waveform (16kHz)
- **Processing**: AST feature extractor creates spectrograms internally
- **Output**: Hidden representations for classification/regression

### Memory Usage
- AST is memory-intensive (requires more GPU memory than ViT)
- Reduce batch size if encountering OOM errors
- Consider gradient checkpointing for large models

## ⚡ Quick Start Summary

```bash
# 1. Download MIT AST model locally
python download_mit_ast.py

# 2. Upload mit-ast-model-for-kaggle.zip to Kaggle as dataset

# 3. Update notebook configuration
AST_MODEL_NAME = '/kaggle/input/your-dataset-name/mit-ast-model-for-kaggle'

# 4. Add dataset to notebook inputs

# 5. Run notebook - it will load MIT AST from local path! 🎉
```

## 🆚 Comparison with ViT Approach

| Aspect | MIT AST | Vision Transformer |
|--------|---------|-------------------|
| Input | Raw audio (16kHz) | Mel-spectrograms (22kHz) |
| Processing | Internal feature extraction | Manual spectrogram creation |
| Architecture | Audio-specific transformer | Vision transformer adapted |
| Memory | Higher GPU memory usage | Lower memory requirements |
| Performance | Optimized for audio | Requires audio→image conversion |

This approach provides a reliable way to use MIT's state-of-the-art AST model for music emotion prediction without network dependency issues in Kaggle environments.
