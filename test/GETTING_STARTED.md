# 🚀 Getting Started with Emotion Prediction Tests

This guide will help you set up and run the emotion prediction test suite.

## 📋 Prerequisites

- Python 3.8+ (preferably 3.11)
- Virtual environment (`.venv` in project root)
- Trained models in `selected/final_best_vit/`
- DEAM dataset in `dataset/DEAM/MEMD_audio/`

## 🔧 Installation Steps

### Step 1: Activate Virtual Environment

```bash
cd /mnt/sdb8mount/free-explore/class/ai/datasets/sentio
source .venv/bin/activate
```

### Step 2: Install PyTorch (if not already installed)

Choose ONE option based on your hardware:

**Option A: CPU Only (Recommended for testing)**
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**Option B: CUDA GPU**
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Step 3: Install Dependencies

```bash
# Install from requirements.txt
pip install -r requirements.txt

# Or install test dependencies specifically
cd test
./install_deps.sh
```

### Step 4: Verify Installation

```bash
python -c "import torch; import librosa; import numpy; print('✅ Core dependencies OK')"
python -c "import transformers; import PIL; print('✅ Test dependencies OK')"
```

## ✅ Verify Models and Data

### Check Models

```bash
ls -lh selected/final_best_vit/
# Should show:
# - best_model.pth
# - mobile_vit_student.pth
# - mobile_vit_student.pt
```

### Check Audio Files

```bash
ls dataset/DEAM/MEMD_audio/*.mp3 | head -5
# Should show audio files like: 10.mp3, 100.mp3, etc.
```

## 🎵 Running Tests

### Quick Test (Recommended First)

```bash
cd test
python quick_test.py
```

Expected output:
```
============================================================
🎵 Audio File: 10.mp3
============================================================

📊 Emotion Predictions:

  Valence: +0.XXXX  (normalized: 0.XXXX)
  Arousal: +0.XXXX  (normalized: 0.XXXX)

Interpretation:
  - Valence: [Positive/Negative/Neutral]
  - Arousal: [High/Low Energy]
  - Overall: [Emotion Quadrant]
============================================================
```

### Single File Prediction

```bash
python predict.py --audio_file ../dataset/DEAM/MEMD_audio/10.mp3
```

**Options:**
- `--model best_vit` - Use main ViT model (default)
- `--model mobile_vit` - Use mobile/distilled model (faster)
- `--device cpu` - Force CPU usage
- `--device cuda` - Use GPU if available

### Batch Prediction

```bash
# Process 10 files
python batch_predict.py --audio_dir ../dataset/DEAM/MEMD_audio --n_samples 10

# Process all files
python batch_predict.py --audio_dir ../dataset/DEAM/MEMD_audio

# Save results
python batch_predict.py --audio_dir ../dataset/DEAM/MEMD_audio --n_samples 20 --format both
```

Results will be saved in `test/results/` as CSV and JSON files.

## 📊 Understanding Output

### Valence (Emotional Positivity)
- `-1.0` = Very Negative (sad, depressed)
- `0.0` = Neutral
- `+1.0` = Very Positive (happy, joyful)

### Arousal (Energy Level)
- `-1.0` = Very Calm (relaxed, peaceful)
- `0.0` = Moderate Energy
- `+1.0` = Very Energetic (excited, intense)

### Emotion Quadrants

| Valence | Arousal | Emotion | Example Genre |
|---------|---------|---------|---------------|
| High (+) | High (+) | Happy/Excited | Pop, Dance |
| High (+) | Low (-) | Calm/Peaceful | Classical, Ambient |
| Low (-) | Low (-) | Sad/Depressed | Blues, Slow Ballad |
| Low (-) | High (+) | Angry/Tense | Metal, Rock |

## 🐛 Troubleshooting

### Issue 1: `ModuleNotFoundError: No module named 'transformers'`

**Solution:**
```bash
source .venv/bin/activate
pip install transformers>=4.30.0 pillow>=10.0.0
```

### Issue 2: `ModuleNotFoundError: No module named 'torch'`

**Solution:**
```bash
# For CPU (recommended)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# For CUDA GPU
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Issue 3: `OSError: libtorch_global_deps.so not found`

**Solution:**
```bash
# Reinstall PyTorch
pip uninstall torch torchaudio
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Issue 4: Model not found

**Solution:**
```bash
# Check if models are extracted
ls -lh selected/final_best_vit/

# If not, extract the zip
cd selected
unzip final_best_vit.zip
```

### Issue 5: Audio file not found

**Solution:**
```bash
# Check dataset location
ls -lh dataset/DEAM/MEMD_audio/ | head

# Adjust path in command
python predict.py --audio_file /absolute/path/to/audio.mp3
```

### Issue 6: CUDA Out of Memory

**Solution:**
```bash
# Use CPU instead
python predict.py --audio_file song.mp3 --device cpu

# Or use mobile model (smaller)
python predict.py --audio_file song.mp3 --model mobile_vit
```

## 📝 Example Workflow

### Complete Test Workflow

```bash
# 1. Setup
cd /mnt/sdb8mount/free-explore/class/ai/datasets/sentio
source .venv/bin/activate

# 2. Install dependencies (first time only)
pip install -r requirements.txt

# 3. Run quick test
cd test
python quick_test.py

# 4. Test single file
python predict.py --audio_file ../dataset/DEAM/MEMD_audio/100.mp3

# 5. Batch process 20 files
python batch_predict.py --audio_dir ../dataset/DEAM/MEMD_audio --n_samples 20

# 6. View results
cat results/predictions_*.csv
```

## 🔍 Advanced Usage

### Use in Python Script

```python
import sys
sys.path.append('test')

from predict import EmotionPredictor

# Initialize
predictor = EmotionPredictor(
    model_path='selected/final_best_vit/best_model.pth',
    model_type='best_vit'
)

# Predict
result = predictor.predict('dataset/DEAM/MEMD_audio/10.mp3')
print(f"Valence: {result['valence']:.4f}")
print(f"Arousal: {result['arousal']:.4f}")
```

### Compare Models

```bash
# Test with ViT
python predict.py --audio_file song.mp3 --model best_vit

# Test with MobileViT
python predict.py --audio_file song.mp3 --model mobile_vit
```

### Custom Audio Files

```bash
# Any supported audio format
python predict.py --audio_file /path/to/your/music.mp3
python predict.py --audio_file /path/to/your/music.wav
python predict.py --audio_file /path/to/your/music.flac
```

## 📚 Documentation

- **Test Suite README**: `test/README.md`
- **Implementation Summary**: `test/IMPLEMENTATION_SUMMARY.md`
- **API Examples**: `test/examples.py`
- **Training Notebook**: `ast/vit_with_gans_emotion_prediction.ipynb`

## ✅ Verification Checklist

- [ ] Virtual environment activated
- [ ] PyTorch installed and working
- [ ] Transformers library installed
- [ ] Librosa and audio dependencies installed
- [ ] Models extracted in `selected/final_best_vit/`
- [ ] DEAM dataset available in `dataset/DEAM/MEMD_audio/`
- [ ] Quick test runs successfully
- [ ] Single file prediction works
- [ ] Batch prediction works
- [ ] Results saved correctly

## 🎯 Next Steps

1. **Run quick test** to verify everything works
2. **Test with sample files** to understand output format
3. **Process your own music** to predict emotions
4. **Integrate into your application** using the Python API
5. **Experiment with different models** to compare performance

## 💡 Tips

- Always activate the virtual environment before running tests
- Use `quick_test.py` to verify setup
- Start with CPU mode if you encounter GPU issues
- Use `mobile_vit` model for faster inference
- Check `results/` directory for saved predictions
- Read code comments in Python files for detailed documentation

## 🤝 Need Help?

1. Check the troubleshooting section above
2. Read the full documentation in `test/README.md`
3. Look at examples in `test/examples.py`
4. Check the implementation summary in `test/IMPLEMENTATION_SUMMARY.md`

## 📊 Expected Performance

- **Single File**: ~2-5 seconds (CPU), ~0.5-1 second (GPU)
- **Batch (10 files)**: ~20-50 seconds (CPU), ~5-10 seconds (GPU)
- **Mobile Model**: ~50% faster than full ViT
- **Accuracy**: CCC ~0.75-0.85 on validation set

Happy testing! 🎵
