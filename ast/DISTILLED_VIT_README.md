# 🎵 Distilled ViT Notebook - Quick Reference

## Overview

A streamlined, production-ready notebook for music emotion recognition using Vision Transformers with GAN-based data augmentation.

**File**: `distilled_vit.ipynb`

## Key Differences from Original

| Feature | Original (`vit_with_gans_emotion_prediction.ipynb`) | Distilled (`distilled_vit.ipynb`) |
|---------|---------------------------------------------------|----------------------------------|
| **Cells** | 80+ cells | 14 cells |
| **Code Style** | Verbose with extensive logging | Concise and focused |
| **Training** | 24 ViT epochs | 30 ViT epochs (more training) |
| **Documentation** | Detailed explanations throughout | Brief markdown headers only |
| **Visualizations** | Multiple intermediate plots | Final results only |
| **Optional Features** | Quality evaluation, audio reconstruction | Removed (core pipeline only) |
| **Error Handling** | Extensive try-catch blocks | Minimal (assumes valid data) |
| **Memory Management** | Explicit cleanup steps | Automatic with key checkpoints |

## 📋 Notebook Structure

The notebook is organized into 18 streamlined cells:

### 1. **Setup & Configuration**
- Install dependencies
- Import libraries
- Set random seeds for reproducibility
- Configure device (GPU/CPU)
- Define paths and hyperparameters

### 2. **Load DEAM Dataset**

### 3. **GAN Architecture** (Code)
- Defines Generator (noise → spectrograms)
- Defines Discriminator (spectrograms → real/fake)

### 4. **Train GAN** (Code)
- 10 epochs of adversarial training
- Alternating discriminator/generator updates
- Progress logging per epoch

### 5. **Generate Synthetic Data** (Code)
- Creates 3200 synthetic spectrograms
- Combines with real data
- Cleans up GAN models to free memory

### 6. **ViT Dataset & DataLoader** (Code)
- Custom PyTorch Dataset with ViT preprocessing
- 70% train / 15% val / 15% test split
- DataLoader setup with memory optimization

### 7. **ViT Model** (Code)
- ViTEmotionModel class
- Pre-trained ViT backbone + regression head
- 2 outputs: valence & arousal

### 8. **Training Setup** (Code)
- CCC metric function
- MSE loss, AdamW optimizer
- CosineAnnealing scheduler

### 9. **Train ViT** (Code)
- 30 epochs of training
- Validation after each epoch
- Best model checkpoint saving
- Progress: Train Loss, Val Loss, CCC metrics

### 10. **Evaluate & Visualize** (Code - Part 1)
- Loads best model
- Evaluates on test set
- Prints final metrics (MSE, MAE, CCC)

### 11. **Visualizations** (Code - Part 2)
- Training/validation loss curves
- CCC scores over time
- Prediction scatter plots

### 12. **Knowledge Distillation - MobileViT Student** (Code)
- MobileViTBlock architecture (efficient attention)
- MobileViTStudent model (~5-8M parameters)
- 10-15x compression vs teacher model

### 13. **Distillation Loss & Training** (Code)
- KnowledgeDistillationLoss (response + feature + attention transfer)
- Feature extraction from teacher/student
- 10 epochs distillation training

### 14. **Teacher vs Student Comparison** (Code)
- Evaluate both models on test set
- Compare metrics (CCC, MAE, model size)
- Performance retention analysis

### 15. **Pipeline Summary** (Markdown)
- Complete workflow recap
- Model file locations
- Deployment guidance

### 16-18. **Results & Diagnostics**
- Summary of completed steps
- Output file locations

## ⏱️ Execution Time

**Total Runtime**: ~90-110 minutes on GPU
- DEAM loading: ~5 min
- GAN training (10 epochs): ~25-30 min
- Synthetic generation: ~5 min
- ViT training (30 epochs): ~45-55 min
- Distillation training (10 epochs): ~10-15 min

**Requirements**: CUDA GPU with 8GB+ VRAM recommended

## Key Hyperparameters

```python
# Audio
SAMPLE_RATE = 22050
DURATION = 30
N_MELS = 128

# GAN
GAN_EPOCHS = 10
GAN_BATCH = 24
GAN_LR = 0.0002
NUM_SYNTHETIC = 3200
LATENT_DIM = 100

# ViT
VIT_EPOCHS = 30        # +25% more than original
VIT_BATCH = 12
VIT_LR = 1e-4
VIT_IMAGE_SIZE = 224
```

## Expected Results

### Performance Metrics
- **Valence CCC**: 0.45 - 0.65 (higher is better)
- **Arousal CCC**: 0.40 - 0.60
- **Test MSE**: 0.10 - 0.20 (lower is better)

### Output Files
```
### Output Files
```
best_vit_model.pth              # Teacher model (~350 MB)
mobile_vit_student.pth          # Student model (~25-40 MB, Android-ready)
generator.pth                   # GAN generator
discriminator.pth               # GAN discriminator
```
```

## Advantages

✅ **Faster execution**: Removed optional quality checks and verbose logging  
✅ **More training**: 30 epochs vs 24 (better convergence)  
✅ **Cleaner code**: No redundant sections or experimental features  
✅ **Production-ready**: Core pipeline only, no debugging artifacts  
✅ **Memory-efficient**: Strategic cleanup between stages  
✅ **Easy to modify**: Compact structure, clear flow  

## When to Use Which Notebook

### Use `distilled_vit.ipynb` when:
- You want quick results
- You understand the pipeline already
- You're doing production runs
- Memory is limited
- You need a clean baseline

### Use `vit_with_gans_emotion_prediction.ipynb` when:
- You're learning the methodology
- You need detailed explanations
- You want to inspect GAN quality
- You want to hear synthetic audio
- You're debugging or experimenting

## Quick Start

```python
# Just run all cells sequentially!
# No configuration needed if using Kaggle paths

# To modify training duration:
VIT_EPOCHS = 50  # Increase for better results
GAN_EPOCHS = 15  # More GAN training

# To reduce memory usage:
VIT_BATCH = 8    # Smaller batch size
NUM_SYNTHETIC = 2000  # Fewer synthetic samples
```

## Customization Guide

### Change Dataset
```python
# In Cell 2 (Load DEAM Dataset)
AUDIO_DIR = '/path/to/your/audio/'
df_annotations = pd.read_csv('/path/to/annotations.csv')
```

### Change Model
```python
# In Cell 7 (ViT Model)
VIT_MODEL_NAME = 'google/vit-base-patch16-224'  # Or any ViT variant
```

### Add Early Stopping
```python
# In Cell 9 (Train ViT), add after validation:
if val_loss < best_val_loss:
    best_val_loss = val_loss
    patience_counter = 0
else:
    patience_counter += 1
    if patience_counter >= 5:
        print("Early stopping triggered")
        break
```

### Export for Mobile
```python
# After training, add a new cell:
import torch.quantization as quantization

# Quantize model for mobile deployment
model.eval()
quantized_model = quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)
torch.save(quantized_model.state_dict(), 'mobile_model.pth')
```

## Troubleshooting

### Out of Memory (OOM)
```python
# Reduce batch sizes
VIT_BATCH = 8
GAN_BATCH = 16

# Reduce synthetic samples
NUM_SYNTHETIC = 2000

# Enable gradient checkpointing (add to ViTEmotionModel)
self.vit.gradient_checkpointing_enable()
```

### Poor Performance
```python
# Increase training epochs
VIT_EPOCHS = 50

# Reduce learning rate
VIT_LR = 5e-5

# Add more synthetic data
NUM_SYNTHETIC = 5000
```

### Model Not Loading
```python
# Fallback to online download
VIT_MODEL_NAME = 'google/vit-base-patch16-224-in21k'

# Or use smaller model
VIT_MODEL_NAME = 'google/vit-base-patch16-224'
```

## Validation Checks

After running, verify:
- ✅ Train loss decreases steadily
- ✅ Val loss follows train loss (no huge gap = no overfitting)
- ✅ CCC values > 0.3 (higher is better)
- ✅ Scatter plots show correlation (not random cloud)
- ✅ Best model saved successfully

## Next Steps

1. **Ensemble**: Train multiple models with different seeds, average predictions
2. **Knowledge Distillation**: Compress to MobileViT (see original notebook)
3. **Data Augmentation**: Add time/frequency masking to ViTDataset
4. **Fine-tuning**: Unfreeze more ViT layers for better performance
5. **Cross-validation**: Split into 5 folds for robust evaluation

---

**Created**: From streamlined version of `vit_with_gans_emotion_prediction.ipynb`  
**Maintained by**: Same codebase as original  
**License**: Follow DEAM dataset license terms
