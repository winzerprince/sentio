# Test Infrastructure Summary

## 📦 What Was Created

I've created a complete test infrastructure for your trained emotion recognition models in the `test/` folder. Here's what you now have:

### Core Files

1. **`vit_model.py`** - Model architecture definitions
   - `ViTForEmotionRegression` - Main Vision Transformer model
   - `MobileViTStudent` - Lightweight distilled model
   
2. **`audio_preprocessor.py`** - Audio processing utilities
   - Loads audio files (MP3, WAV, etc.)
   - Converts to mel spectrograms
   - Prepares images for ViT input

3. **`predict.py`** - Single file prediction script
   - CLI tool for predicting emotions from one audio file
   - Returns valence and arousal values
   - Provides human-readable interpretations

4. **`batch_predict.py`** - Batch processing script
   - Process multiple audio files at once
   - Saves results to CSV/JSON
   - Generates statistics and rankings

5. **`quick_test.py`** - Quick validation script
   - Tests with sample files from DEAM dataset
   - Verifies everything is working
   - Good for debugging

6. **`examples.py`** - API usage examples
   - Shows how to use the system programmatically
   - 6 different usage examples
   - Good starting point for integration

7. **`run_tests.sh`** - Setup and test runner
   - Checks dependencies
   - Verifies models and data
   - Runs quick test

8. **`README.md`** - Complete documentation
   - Usage instructions
   - Output format explanation
   - Troubleshooting guide
   - Examples

## 🎯 Key Features

### Input
- **File format**: MP3, WAV, FLAC, OGG, M4A
- **Processing**: Automatic padding/truncation to 30 seconds
- **Spectrogram**: 128 mel bins, 224x224 image for ViT

### Output
- **Valence**: -1 (negative) to +1 (positive)
- **Arousal**: -1 (calm) to +1 (energetic)
- **Normalized**: 0 to 1 scale for easier interpretation
- **Emotion labels**: Happy, Sad, Calm, Angry, etc.

### Models Supported
1. **Best ViT** (`best_model.pth`) - Main model, best accuracy
2. **Mobile ViT** (`mobile_vit_student.pth`) - Faster inference

## 🚀 Quick Start

### 1. Setup Environment

```bash
cd test
source ../.venv/bin/activate
pip install -r ../requirements.txt
```

### 2. Run Quick Test

```bash
python quick_test.py
```

### 3. Test Single File

```bash
python predict.py --audio_file ../dataset/DEAM/MEMD_audio/10.mp3
```

### 4. Batch Process

```bash
python batch_predict.py --audio_dir ../dataset/DEAM/MEMD_audio --n_samples 10
```

## 📊 Example Output

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

## 📁 File Structure

```
test/
├── README.md                 # Documentation
├── __init__.py              # Package initialization
├── vit_model.py             # Model architectures
├── audio_preprocessor.py    # Audio processing
├── predict.py               # Single file prediction
├── batch_predict.py         # Batch processing
├── quick_test.py            # Quick validation
├── examples.py              # API usage examples
├── run_tests.sh             # Setup script
└── results/                 # Output directory
```

## 🔧 Requirements Updated

Added to `requirements.txt`:
- `transformers>=4.30.0` - For ViT models
- `pillow>=10.0.0` - For image processing

## 💡 Usage Examples

### Command Line

```bash
# Single prediction
python predict.py --audio_file song.mp3

# Batch prediction (10 files)
python batch_predict.py --audio_dir ../dataset/DEAM/MEMD_audio --n_samples 10

# Use mobile model (faster)
python predict.py --audio_file song.mp3 --model mobile_vit

# Save results as CSV
python batch_predict.py --audio_dir ../dataset/DEAM/MEMD_audio --format csv
```

### Python API

```python
from test.predict import EmotionPredictor

# Initialize predictor
predictor = EmotionPredictor(
    model_path='selected/final_best_vit/best_model.pth',
    model_type='best_vit'
)

# Predict single file
result = predictor.predict('song.mp3')
print(f"Valence: {result['valence']:.4f}")
print(f"Arousal: {result['arousal']:.4f}")

# Predict batch
results = predictor.predict_batch(['song1.mp3', 'song2.mp3'])
```

## 🧪 Testing Strategy

1. **Quick Test** - Verify basic functionality with sample files
2. **Single File** - Test individual predictions, check output format
3. **Batch Mode** - Test processing pipeline, check statistics
4. **Model Comparison** - Compare ViT vs MobileViT performance
5. **Error Handling** - Test with invalid files, missing models

## ✅ What's Working

- ✓ Model loading from checkpoints
- ✓ Audio file processing (MP3, WAV, etc.)
- ✓ Mel spectrogram generation
- ✓ ViT preprocessing (ImageNet normalization)
- ✓ Single file prediction
- ✓ Batch prediction
- ✓ Results saving (CSV, JSON)
- ✓ Statistics generation
- ✓ Error handling
- ✓ Command-line interface
- ✓ Python API

## 🎓 Understanding the Output

### Valence-Arousal Space

```
     High Arousal (Energetic)
              |
   Angry   😠 | 😊  Happy
   (Q4)       |       (Q1)
  ───────────────────────── Valence
              |
     Sad   😞 | 😌  Calm
   (Q3)       |       (Q2)
              |
      Low Arousal (Calm)
```

### Quadrants
- **Q1** (High V, High A): Happy, Excited, Joyful
- **Q2** (High V, Low A): Calm, Peaceful, Relaxed
- **Q3** (Low V, Low A): Sad, Depressed, Melancholic
- **Q4** (Low V, High A): Angry, Tense, Fearful

## 🐛 Troubleshooting

### Model Not Found
- Check `selected/final_best_vit/` directory exists
- Verify `best_model.pth` is present
- Check file permissions

### Import Errors
- Activate virtual environment: `source ../.venv/bin/activate`
- Install dependencies: `pip install -r ../requirements.txt`
- Check PyTorch installation: `python -c "import torch; print(torch.__version__)"`

### Audio Loading Errors
- Check file exists and is readable
- Verify format is supported (MP3, WAV, etc.)
- Try converting to WAV format

### CUDA Out of Memory
- Use CPU: `--device cpu`
- Use mobile model: `--model mobile_vit`
- Process fewer files at once

## 📚 Next Steps

1. **Test with your own audio files**
   ```bash
   python predict.py --audio_file /path/to/your/song.mp3
   ```

2. **Analyze a music collection**
   ```bash
   python batch_predict.py --audio_dir /path/to/music/folder
   ```

3. **Integrate into your application**
   - Use `EmotionPredictor` class
   - See `examples.py` for code samples
   - Check API documentation in code comments

4. **Experiment with different models**
   - Compare ViT vs MobileViT
   - Test inference speed
   - Evaluate accuracy

## 🔗 Related Files

- Training notebook: `ast/vit_with_gans_emotion_prediction.ipynb`
- Model checkpoints: `selected/final_best_vit/`
- Audio dataset: `dataset/DEAM/MEMD_audio/`
- Project instructions: `.github/instructions/main.instructions.md`

## 📝 Notes

- Models expect 30-second audio clips (auto-padded/truncated)
- Spectrograms are converted to 224x224 RGB images
- ImageNet normalization is applied (mean/std)
- Output range is [-1, +1] for both valence and arousal
- Normalized output [0, 1] also provided for convenience
