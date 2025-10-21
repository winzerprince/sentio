# Quick Start Guide - Improved ViT + GAN Notebook

## 🚀 Running the Notebook on Kaggle

### Prerequisites
1. Kaggle account with GPU enabled
2. DEAM dataset uploaded as Kaggle dataset
3. Pre-trained ViT model uploaded (optional, but recommended)

### Step-by-Step Instructions

#### 1. Upload to Kaggle
- Go to kaggle.com and create a new notebook
- Upload `vit_with_gans_emotion_prediction.ipynb`
- Enable GPU: Settings → Accelerator → GPU T4 x2 (or P100)

#### 2. Add Datasets
Add the following datasets to your notebook:
- DEAM audio files and annotations
- (Optional) Pre-trained ViT model to avoid download issues

#### 3. Update Paths
Check these configuration variables in cell 2:
```python
AUDIO_DIR = '/kaggle/input/deam-mediaeval-dataset-emotional-analysis-in-music/DEAM_audio/MEMD_audio/'
ANNOTATIONS_DIR = '/kaggle/input/deam-mediaeval-dataset-emotional-analysis-in-music/DEAM_Annotations/annotations/annotations averaged per song/song_level/'
VIT_MODEL_NAME = '/kaggle/input/vit-model-kaggle/vit-model-for-kaggle'
```

#### 4. Run All Cells
- Click "Run All" or run cells sequentially
- **Estimated time**: 45-60 minutes on Kaggle GPU

---

## 📊 What to Expect

### GAN Training (10 epochs, ~10-15 minutes)
**Look for:**
- Discriminator accuracy stabilizing around 70-80%
- Generator/Discriminator loss ratio close to 1.0
- Quality score above 50

**Warning signs:**
- Discriminator accuracy stuck at 50% (mode collapse)
- Discriminator accuracy at 100% (discriminator too strong)
- Quality score below 30 (mostly noise)

### ViT Training (24 epochs, ~30-40 minutes)
**Look for:**
- Validation loss decreasing smoothly
- CCC (Concordance Correlation) above 0.5
- Progressive unfreezing messages at epochs 8, 12, 16

**Warning signs:**
- Validation loss increasing (overfitting)
- Very large gap between train and validation loss
- CCC below 0.3 (poor correlation)

---

## 🎛️ Key Hyperparameters to Tune

### If GAN generates noise:
```python
GAN_EPOCHS = 15  # Increase from 10
GAN_LR = 0.0001  # Decrease from 0.0002
```

### If ViT overfits:
```python
DROPOUT = 0.2  # Increase from 0.1
WEIGHT_DECAY = 0.1  # Increase from 0.05
```

### If training is slow:
```python
BATCH_SIZE = 8  # Decrease from 16
NUM_EPOCHS = 20  # Decrease from 24
```

### If out of memory:
```python
BATCH_SIZE = 8  # Decrease
GAN_BATCH_SIZE = 16  # Decrease from 32
```

---

## 📈 Interpreting Results

### GAN Quality Metrics
| Score | Quality | What it means |
|-------|---------|---------------|
| 70-100 | Excellent | Spectrograms look realistic |
| 50-69 | Good | Acceptable for augmentation |
| 30-49 | Fair | Marginal improvement over noise |
| 0-29 | Poor | Not useful, retrain GAN |

### ViT Performance Metrics
| Metric | Good | Acceptable | Poor |
|--------|------|------------|------|
| MAE | < 0.25 | 0.25-0.35 | > 0.35 |
| CCC Valence | > 0.6 | 0.4-0.6 | < 0.4 |
| CCC Arousal | > 0.5 | 0.3-0.5 | < 0.3 |

---

## 🎧 Listening to Generated Audio

After training, the notebook generates 5 audio samples. Evaluate them:

### What to listen for:
- ✅ **Tonal structure**: Does it sound like music or white noise?
- ✅ **Emotion variation**: Do happy samples sound different from sad ones?
- ✅ **Temporal patterns**: Are there rhythmic or melodic patterns?

### What's acceptable:
- Audio may sound distorted or robotic (this is normal)
- Some samples may be better than others
- Clear emotional differences are more important than audio quality

### Red flags:
- All samples sound identical
- Pure white noise with no structure
- Extreme clipping or distortion

---

## 🔧 Troubleshooting

### Problem: NameError - train_loader not defined
**Solution:** ✅ This is fixed in the improved notebook. Section 7 creates the dataloaders.

### Problem: GAN loss explodes (NaN values)
**Solution:** 
- Reduce GAN_LR to 0.0001
- Check if spectrograms have NaN values
- Ensure gradient clipping is enabled

### Problem: Discriminator always wins (accuracy = 100%)
**Solution:**
- Lower discriminator learning rate (try 0.0001)
- Increase GAN_EPOCHS to 15-20
- The adaptive training should handle this, but may need more time

### Problem: Generator always loses (accuracy = 0%)
**Solution:**
- Increase generator training steps (modify code: `g_steps = 2`)
- Check if spectrograms are normalized correctly
- Reduce discriminator complexity

### Problem: ViT performance is poor (CCC < 0.3)
**Solution:**
- Ensure ViT model loaded correctly (check for warnings)
- Increase NUM_EPOCHS to 30-35
- Try unfreezing more layers earlier
- Check if data normalization is correct (ImageNet stats)

### Problem: Out of memory
**Solution:**
```python
BATCH_SIZE = 8  # Reduce
GAN_BATCH_SIZE = 16  # Reduce
NUM_SYNTHETIC = 2000  # Reduce from 3200
```

### Problem: Training too slow
**Solution:**
- Ensure GPU is enabled in Kaggle settings
- Reduce NUM_EPOCHS to 20
- Use mixed precision (already enabled)
- Reduce GAN_EPOCHS to 8

---

## 📁 Output Files

After training, these files are created in `/kaggle/working/vit_augmented/`:

### Models:
- `improved_generator.pth` - Trained GAN generator
- `improved_discriminator.pth` - Trained GAN discriminator  
- `best_vit_model.pth` - Best ViT model checkpoint

### Visualizations:
- `gan_balanced_training_curves.png` - GAN training metrics
- `gan_quality_evaluation.png` - Quality assessment plots
- `generated_sample_*.wav` - Audio samples from GAN
- `generated_spec_*.png` - Spectrograms of GAN samples
- `vit_training_curves.png` - ViT training metrics
- `prediction_scatter.png` - Predicted vs actual emotions
- `va_space_comparison.png` - Valence-arousal space

### Download them:
```python
# At the end of the notebook, this creates a zip
!zip -r /kaggle/working/vit_output.zip /kaggle/working/vit_augmented
```

---

## 🎯 Expected Results

### Baseline (without improvements):
- GAN Quality Score: ~25-35
- ViT MAE: ~0.35-0.40
- ViT CCC: ~0.35-0.45
- Training stability: Unstable, mode collapse common

### With improvements (this notebook):
- GAN Quality Score: ~50-70
- ViT MAE: ~0.25-0.30
- ViT CCC: ~0.50-0.65
- Training stability: Stable, balanced training

### State-of-the-art (for reference):
- MAE: ~0.18-0.22
- CCC: ~0.65-0.75
- Requires: More data, ensemble models, specialized architectures

---

## 📚 Understanding the Improvements

### 1. Self-Attention in GAN
**What it does:** Captures long-range patterns in spectrograms (like how a chorus relates to verses)

**Why it helps:** Music has structure across time, not just local patterns

### 2. Balanced Training
**What it does:** Adjusts training frequency so neither network dominates

**Why it helps:** Prevents mode collapse and ensures stable convergence

### 3. Progressive Unfreezing
**What it does:** Gradually adapts pretrained ViT to audio domain

**Why it helps:** Preserves useful ImageNet features while learning audio-specific patterns

### 4. Weighted Loss
**What it does:** Prioritizes valence over arousal (60/40 split)

**Why it helps:** Valence is often more reliably annotated and perceptually salient

### 5. Audio Reconstruction
**What it does:** Converts spectrograms back to audio using Griffin-Lim

**Why it helps:** Allows qualitative assessment - you can hear if GAN output makes sense

---

## 🎓 Next Experiments to Try

### Easy (1-2 hour experiments):
1. Try different valence/arousal weights (0.5/0.5, 0.7/0.3)
2. Adjust progressive unfreezing schedule (unfreeze earlier/later)
3. Increase GAN epochs to 15-20
4. Try different batch sizes

### Medium (half-day experiments):
1. Add more data augmentation (time-stretch, pitch-shift)
2. Use different ViT model (vit-large, swin-transformer)
3. Implement multi-scale discriminator
4. Add perceptual loss using VGGish

### Hard (multi-day experiments):
1. Replace GAN with diffusion model
2. Use transformer-based generator
3. Add contrastive learning pre-training
4. Implement attention rollout visualization

---

## 💡 Pro Tips

1. **Save intermediate results**: The notebook saves models after training, but consider saving after each major phase

2. **Monitor training actively**: Don't just run all cells - watch the metrics to catch issues early

3. **Use version control**: Save different versions as you tune hyperparameters

4. **Document experiments**: Keep a log of what hyperparameters you tried and results

5. **Start small**: If testing changes, reduce epochs/samples first to iterate faster

6. **Compare baselines**: Keep a baseline notebook to compare against improvements

---

## 🆘 Getting Help

If you encounter issues not covered here:

1. Check the full summary document: `VIT_GAN_IMPROVEMENTS_SUMMARY.md`
2. Review the error message carefully - often it tells you exactly what's wrong
3. Check Kaggle discussions for similar issues
4. Ensure all paths are correct for Kaggle environment

---

## ✅ Success Checklist

Before considering the experiment successful:

- [ ] GAN trains without errors
- [ ] GAN quality score > 50
- [ ] Discriminator accuracy between 60-85%
- [ ] Generated audio has some musical structure
- [ ] ViT trains without errors
- [ ] Validation loss decreases over time
- [ ] CCC > 0.5 for both valence and arousal
- [ ] No extreme overfitting (val loss << train loss)
- [ ] All output files generated successfully

---

Good luck with your experiments! 🎵🎨🤖
