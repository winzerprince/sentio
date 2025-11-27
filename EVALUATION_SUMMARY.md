# Model Evaluation Summary

## 🎯 Quick Results

**Evaluated**: 20 songs from DEAM dataset  
**Model**: Vision Transformer (ViT) for emotion regression  
**Date**: October 29, 2025

### Performance Scores

| Metric | Valence | Arousal | Average |
|--------|---------|---------|---------|
| **MAE** | 0.22 | 0.26 | **0.24** |
| **Correlation** | 0.32 | 0.50 | **0.41** |
| **R² Score** | -0.11 | 0.22 | **0.06** |

### Interpretation

✅ **Arousal (Energy)**: Moderate performance  
- 50% correlation with ground truth
- Model explains 22% of variance
- Better at detecting energetic vs calm music

⚠️ **Valence (Positive/Negative)**: Needs improvement  
- 32% correlation with ground truth  
- Struggles to predict positive vs negative emotions
- This is a known challenge in music emotion recognition

📊 **Overall**: Average error of ±0.24 on [-1, +1] scale (±24%)

## 📂 Files Generated

All evaluation results are in `test/evaluation_results/`:

- **evaluation_results.csv** - Detailed predictions for each song
- **metrics.txt** - Performance metrics
- **scatter_plots.png** - Prediction accuracy visualization  
- **error_distribution.png** - Error analysis histograms
- **emotion_space.png** - 2D emotion space (valence × arousal)
- **prediction_comparison.png** - Visual error comparison with arrows

## 🚀 How to Run

```bash
cd test
python evaluate_model.py --n_samples 20 --model_path ../selected/final_best_vit/best_model.pth
```

For more samples:
```bash
python evaluate_model.py --n_samples 50 --model_path ../selected/final_best_vit/best_model.pth
```

## 📊 Sample Predictions

| Song | True Valence | Predicted | Error | True Arousal | Predicted | Error |
|------|--------------|-----------|-------|--------------|-----------|-------|
| 1293 | +0.18 | +0.18 | ✓ 0.00 | +0.02 | +0.13 | 0.11 |
| 1000 | +0.45 | +0.10 | 0.35 | +0.15 | -0.07 | 0.22 |
| 503  | +0.20 | +0.20 | ✓ 0.00 | +0.35 | -0.01 | 0.36 |
| 1118 | -0.25 | +0.09 | 0.34 | -0.78 | -0.25 | 0.53 |

## 📖 Documentation

- **Full evaluation details**: `test/EVALUATION_README.md`
- **Test suite documentation**: `test/README.md`
- **Getting started**: `test/GETTING_STARTED.md`

## 🔍 Key Insights

1. **Model shows moderate correlation** with human annotations (0.41 average)
2. **Arousal predictions are more reliable** than valence predictions
3. **Average error is ±24%**, which is reasonable for emotion recognition
4. **Model tends to predict near-neutral** emotions (conservative predictions)
5. **Strong emotional extremes** (very happy/sad) are harder to predict

## 💡 Next Steps

To improve performance:
- [ ] Increase training data size
- [ ] Try ensemble of multiple models
- [ ] Experiment with different audio features
- [ ] Fine-tune on more DEAM annotations
- [ ] Address valence prediction specifically
- [ ] Test on different music genres separately

## 📚 Learn More

- See `test/EVALUATION_README.md` for detailed methodology
- View generated plots in `test/evaluation_results/`
- Check `test/evaluation_results/evaluation_results.csv` for all predictions
