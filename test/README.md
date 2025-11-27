# Emotion Prediction Test Suite

This directory contains scripts for testing trained emotion recognition models on audio files. The models predict **valence** (positivity/negativity) and **arousal** (energy level) from music.

## 📁 Files

- `predict.py` - Single file emotion prediction
- `batch_predict.py` - Batch processing of multiple files
- `vit_model.py` - Model architecture definitions
- `audio_preprocessor.py` - Audio preprocessing utilities
- `quick_test.py` - Quick test script with sample files
- `results/` - Output directory for batch predictions

## 🚀 Quick Start

### 1. Activate Virtual Environment

```bash
source ../.venv/bin/activate
```

### 2. Single File Prediction

Predict emotions for a single audio file:

```bash
python predict.py --audio_file ../dataset/DEAM/MEMD_audio/10.mp3
```

With specific model:

```bash
python predict.py --audio_file ../dataset/DEAM/MEMD_audio/10.mp3 --model best_vit
python predict.py --audio_file ../dataset/DEAM/MEMD_audio/10.mp3 --model mobile_vit
```

### 3. Batch Prediction

Process multiple files from a directory:

```bash
# Process 10 random samples
python batch_predict.py --audio_dir ../dataset/DEAM/MEMD_audio --n_samples 10

# Process all files in directory
python batch_predict.py --audio_dir ../dataset/DEAM/MEMD_audio

# Process specific files
python batch_predict.py --audio_files song1.mp3 song2.mp3 song3.mp3
```

### 4. Quick Test

Run a quick test with predefined samples:

```bash
python quick_test.py
```

## 📊 Output Format

### Single Prediction

```
============================================================
🎵 Audio File: 10.mp3
============================================================

📊 Emotion Predictions:

  Valence: +0.4523  (normalized: 0.7262)
  Arousal: +0.2134  (normalized: 0.6067)

Interpretation:
  - Valence: Positive 🙂
  - Arousal: High Energy 🔥
  - Overall: Happy/Excited (High Valence, High Arousal) 🎉

============================================================
```

### Batch Prediction

Results are saved as CSV and JSON files in `results/` directory:

**CSV Output:**
```csv
file,valence,arousal,valence_normalized,arousal_normalized,valence_interpretation,arousal_interpretation,emotion
10.mp3,0.4523,0.2134,0.7262,0.6067,Positive 🙂,High Energy 🔥,Happy/Excited
```

**Statistics:**
- Summary statistics (mean, median, min, max)
- Emotion quadrant distribution
- Top predictions by valence and arousal

## 🎯 Understanding the Output

### Valence (Emotional Positivity)
- **Range**: -1 (very negative) to +1 (very positive)
- **Interpretation**:
  - `> +0.5`: Very Positive 😊
  - `+0.1 to +0.5`: Positive 🙂
  - `-0.1 to +0.1`: Neutral 😐
  - `-0.5 to -0.1`: Negative 🙁
  - `< -0.5`: Very Negative 😢

### Arousal (Energy Level)
- **Range**: -1 (very calm) to +1 (very energetic)
- **Interpretation**:
  - `> +0.5`: Very High Energy ⚡
  - `+0.1 to +0.5`: High Energy 🔥
  - `-0.1 to +0.1`: Moderate Energy 💫
  - `-0.5 to -0.1`: Low Energy 😌
  - `< -0.5`: Very Low Energy 💤

### Emotion Quadrants

Based on valence-arousal combinations:

| Quadrant | Valence | Arousal | Emotion | Example |
|----------|---------|---------|---------|---------|
| Q1 | Positive | High | Happy/Excited 🎉 | Dance music |
| Q2 | Positive | Low | Calm/Peaceful 😌 | Ambient music |
| Q3 | Negative | Low | Sad/Depressed 😞 | Slow ballad |
| Q4 | Negative | High | Angry/Tense 😠 | Heavy metal |

## 🔧 Command-Line Options

### `predict.py`

```bash
python predict.py --help

Options:
  --audio_file PATH      Path to audio file (required)
  --model MODEL         Model type: best_vit, mobile_vit, vit (default: best_vit)
  --model_dir DIR       Model checkpoint directory (default: ../selected/final_best_vit)
  --device DEVICE       Device: cpu or cuda (auto-detected)
```

### `batch_predict.py`

```bash
python batch_predict.py --help

Options:
  --audio_dir DIR       Directory containing audio files
  --audio_files FILES   List of audio file paths
  --n_samples N         Number of samples to process (default: all)
  --model MODEL         Model type (default: best_vit)
  --model_dir DIR       Model checkpoint directory
  --output_dir DIR      Output directory for results (default: ./results)
  --format FORMAT       Output format: csv, json, both (default: both)
  --device DEVICE       Device: cpu or cuda
```

## 🧪 Available Models

### 1. Best ViT (Recommended)
- **File**: `best_model.pth`
- **Type**: Vision Transformer
- **Use**: `--model best_vit`
- **Accuracy**: Highest
- **Speed**: Moderate

### 2. Mobile ViT (Fast)
- **File**: `mobile_vit_student.pth`
- **Type**: Distilled MobileViT
- **Use**: `--model mobile_vit`
- **Accuracy**: Good
- **Speed**: Fast

## 📦 Model Location

Models are located in: `../selected/final_best_vit/`

Files:
- `best_model.pth` - Main ViT model
- `mobile_vit_student.pth` - Distilled mobile model
- `mobile_vit_student.pt` - Alternative checkpoint
- `mobile_vit_student.onnx` - ONNX format

## 🎵 Supported Audio Formats

- MP3
- WAV
- FLAC
- OGG
- M4A
- Any format supported by librosa

## ⚙️ Technical Details

### Audio Processing
- **Sample Rate**: 22,050 Hz
- **Duration**: 30 seconds (padded/truncated)
- **Mel Bins**: 128
- **FFT Size**: 2048
- **Hop Length**: 512

### Model Input
- **Image Size**: 224x224
- **Channels**: 3 (RGB)
- **Normalization**: ImageNet mean/std

## 🐛 Troubleshooting

### Model Not Found
```
❌ Error: Model not found at ../selected/final_best_vit/best_model.pth
```

**Solution**: Check that models are extracted in the correct location:
```bash
cd ../selected
ls -la final_best_vit/
```

### Audio File Error
```
❌ Failed to load audio from path/to/file.mp3
```

**Solution**: 
- Ensure file exists and is readable
- Check file format is supported
- Try converting to WAV format

### CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```

**Solution**: Use CPU instead:
```bash
python predict.py --audio_file song.mp3 --device cpu
```

### Import Errors
```
ModuleNotFoundError: No module named 'transformers'
```

**Solution**: Install dependencies:
```bash
source ../.venv/bin/activate
pip install -r ../requirements.txt
```

## 📝 Examples

### Example 1: Single Prediction
```bash
python predict.py --audio_file ../dataset/DEAM/MEMD_audio/1000.mp3
```

### Example 2: Batch Process 20 Songs
```bash
python batch_predict.py \
  --audio_dir ../dataset/DEAM/MEMD_audio \
  --n_samples 20 \
  --model best_vit \
  --format both
```

### Example 3: Process Specific Files
```bash
python batch_predict.py \
  --audio_files \
    ../dataset/DEAM/MEMD_audio/10.mp3 \
    ../dataset/DEAM/MEMD_audio/20.mp3 \
    ../dataset/DEAM/MEMD_audio/30.mp3 \
  --output_dir ./my_results
```

### Example 4: Use Mobile Model (Faster)
```bash
python predict.py \
  --audio_file ../dataset/DEAM/MEMD_audio/500.mp3 \
  --model mobile_vit
```

## 🔬 Research & Citation

These models were trained on the DEAM (Database for Emotional Analysis of Music) dataset using Vision Transformers with GAN-based data augmentation.

**Model Performance:**
- Validation CCC: ~0.75-0.85
- Valence MAE: ~0.15
- Arousal MAE: ~0.18

## 📚 Additional Resources

- Training notebook: `../ast/vit_with_gans_emotion_prediction.ipynb`
- Model documentation: `../ast/VIT_QUICKSTART.md`
- Project instructions: `../.github/instructions/main.instructions.md`

## 💡 Tips

1. **First run is slower**: Model needs to be loaded into memory
2. **Batch processing is more efficient**: Use `batch_predict.py` for multiple files
3. **Mobile model for speed**: Use `--model mobile_vit` for faster inference
4. **Save results**: Use `--format both` to save as CSV and JSON
5. **Check statistics**: Batch mode provides useful statistics and rankings

## 🤝 Contributing

To add new model architectures or preprocessing methods:

1. Add model class to `vit_model.py`
2. Update `predict.py` to support new model type
3. Add checkpoint to `../selected/` directory
4. Update this README with usage instructions
