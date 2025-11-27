# Model Evaluation Against DEAM Annotations

## Overview

This evaluation compares our trained Vision Transformer emotion recognition model against the ground truth annotations from the DEAM (Database for Emotional Analysis in Music) dataset.

## Evaluation Setup

- **Model**: Vision Transformer (ViT) fine-tuned for emotion regression
- **Dataset**: DEAM annotations (averaged per song)
- **Sample Size**: 20 songs (randomly selected)
- **Evaluation Metrics**: MAE, RMSE, Correlation, R² Score
- **Dimensions**: Valence (positive/negative) and Arousal (energetic/calm)

## Results Summary

### Performance Metrics

| Metric | Valence | Arousal | Average |
|--------|---------|---------|---------|
| **MAE** | 0.2164 | 0.2558 | 0.2361 |
| **RMSE** | 0.2843 | 0.2925 | 0.2884 |
| **Correlation** | 0.3236 | 0.5033 | 0.4134 |
| **R² Score** | -0.1092 | 0.2218 | 0.0563 |

### Key Findings

1. **Arousal Predictions**: The model performs better on arousal (energy level) with:
   - Moderate positive correlation (0.50)
   - R² score of 0.22 indicates the model explains ~22% of arousal variance
   
2. **Valence Predictions**: More challenging with:
   - Weak positive correlation (0.32)
   - Negative R² suggests model struggles with valence prediction
   
3. **Overall Performance**:
   - Average MAE of 0.24 on a scale of [-1, 1] (±24% error)
   - Average correlation of 0.41 shows moderate alignment with ground truth
   - Model shows promise but has room for improvement

### Interpretation

**What the metrics mean:**

- **MAE (Mean Absolute Error)**: Average difference between prediction and truth
  - Lower is better (0 = perfect)
  - Our 0.24 means predictions are typically off by ±0.24 on [-1, 1] scale
  
- **RMSE (Root Mean Squared Error)**: Similar to MAE but penalizes large errors more
  - Our 0.29 is slightly higher than MAE, indicating some outlier predictions
  
- **Correlation**: How well predictions move in the same direction as truth
  - Range: -1 to +1, where 1 = perfect positive correlation
  - Our 0.41 shows moderate positive relationship
  
- **R² Score**: Proportion of variance explained by the model
  - Range: -∞ to 1, where 1 = perfect predictions
  - Our 0.06 indicates the model explains only 6% of overall variance
  - Negative R² for valence means model performs worse than predicting the mean

## Visualizations

The evaluation generates 4 comprehensive plots:

### 1. Scatter Plots (`scatter_plots.png`)
- **Top Row**: Valence and Arousal predictions vs ground truth
  - Blue/red dots = individual predictions
  - Red dashed line = perfect prediction (45° line)
  - Green line = actual regression fit
- **Bottom Row**: Residual plots showing prediction errors
  - Points near zero = good predictions
  - Spread indicates prediction uncertainty

### 2. Error Distribution (`error_distribution.png`)
- Histograms showing how prediction errors are distributed
- Centered at zero = unbiased predictions
- Width indicates prediction variability

### 3. Emotion Space (`emotion_space.png`)
- **Left**: Ground truth emotions in 2D valence-arousal space
- **Right**: Model predictions in same space
- Shows how emotions cluster in 4 quadrants:
  - Q1 (top-right): Happy/Excited
  - Q2 (top-left): Angry/Tense
  - Q3 (bottom-left): Sad/Depressed
  - Q4 (bottom-right): Calm/Peaceful

### 4. Prediction Comparison (`prediction_comparison.png`)
- Combined view with arrows from ground truth (green) to predictions (red)
- Arrow length = prediction error magnitude
- Arrow direction = error direction in emotion space

## Files Generated

```
evaluation_results/
├── evaluation_results.csv    # Detailed predictions for all songs
├── metrics.txt               # Performance metrics summary
├── scatter_plots.png         # Regression and residual plots
├── error_distribution.png    # Error histograms
├── emotion_space.png         # 2D emotion space visualization
└── prediction_comparison.png # Ground truth vs predictions with error vectors
```

## Usage

### Basic Evaluation (20 songs)
```bash
python evaluate_model.py \
  --n_samples 20 \
  --model_path ../selected/final_best_vit/best_model.pth
```

### Evaluate More Songs
```bash
python evaluate_model.py \
  --n_samples 50 \
  --model_path ../selected/final_best_vit/best_model.pth \
  --output_dir ./evaluation_50_songs
```

### Custom Annotation File
```bash
python evaluate_model.py \
  --annotation_file /path/to/annotations.csv \
  --audio_dir /path/to/audio \
  --model_path ../selected/final_best_vit/best_model.pth
```

## Understanding DEAM Annotations

The DEAM dataset uses a **1-9 scale** for both valence and arousal:
- 1 = Most negative/calm
- 5 = Neutral/moderate
- 9 = Most positive/excited

Our model outputs **-1 to +1 scale**, so we normalize:
```
normalized = (deam_score - 5) / 4
```

This converts:
- DEAM 1 → -1 (most negative/calm)
- DEAM 5 → 0 (neutral)
- DEAM 9 → +1 (most positive/excited)

## Future Improvements

Based on evaluation results, potential improvements:

1. **Data Augmentation**: Increase training data variety
2. **Architecture Tuning**: Experiment with different model architectures
3. **Feature Engineering**: Try different audio feature representations
4. **Multi-Task Learning**: Joint training on related tasks
5. **Ensemble Methods**: Combine multiple model predictions
6. **Valence Focus**: Special attention to valence prediction (currently weak)

## Sample Predictions

Here are some example predictions from the evaluation:

| Song ID | True Valence | Pred Valence | True Arousal | Pred Arousal | Error |
|---------|--------------|--------------|--------------|--------------|-------|
| 1000 | +0.45 | +0.10 | +0.15 | -0.07 | ✓ Similar direction |
| 854 | -0.53 | +0.13 | -0.30 | -0.01 | ✗ Wrong valence sign |
| 1118 | -0.25 | +0.09 | -0.78 | -0.25 | ✓ Good arousal |
| 1293 | +0.18 | +0.18 | +0.02 | +0.13 | ✓✓ Excellent match |

## Notes

- Random seed (42) ensures reproducibility
- Evaluation uses CPU by default (add `--device cuda` for GPU)
- Missing audio files are automatically skipped
- Results saved in CSV format for further analysis

## Citation

If using this evaluation framework, please cite the DEAM dataset:

```
Aljanaki, A., Yang, Y. H., & Soleymani, M. (2017). 
Developing a benchmark for emotional analysis of music. 
PloS one, 12(3), e0173392.
```
