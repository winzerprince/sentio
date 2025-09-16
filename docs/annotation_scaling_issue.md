# Critical Discovery: Annotation Scaling Issue

**Date:** September 16, 2025  
**Issue:** Annotation scaling was not applied during training

## 🚨 Important Finding

During model prediction testing, we discovered that the intended annotation scaling from [1,9] to [-1,1] was **NOT applied** during training, despite being implemented in the code.

## 🔍 Root Cause Analysis

### The Problem
The `DataLoader._scale_annotations()` function in `src/data_processing/data_loader.py` contains this logic:

```python
def _scale_annotations(self, annotations):
    """
    Scales valence and arousal from [1, 9] to [-1, 1].
    """
    logger.info("Scaling annotations from [1, 9] to [-1, 1]")
    for col in ['valence', 'arousal']:  # ❌ Looking for these column names
        if col in annotations.columns:
            annotations[col] = (annotations[col] - 5.0) / 4.0
    return annotations
```

### The Actual Data
The annotation file has these columns:
```csv
song_id, valence_mean, valence_std, arousal_mean, arousal_std
2,3.1,0.94,3,0.63
3,3.5,1.75,3.3,1.62
4,5.7,1.42,5.5,1.63
```

**Mismatch:** 
- Code looks for: `['valence', 'arousal']`
- File contains: `['valence_mean', 'arousal_mean']`
- **Result:** Scaling was never applied!

## 📊 Impact Assessment

### What Actually Happened
1. **Training data:** Models were trained on original [1,9] scale
2. **Performance metrics:** All reported R² scores are valid but based on [1,9] scale
3. **Predictions:** Models output values in [1,9] range, not [-1,1]

### Model Performance (Still Valid)
- **XGBoost**: R² = 0.540 (overall best)
- **SVR**: R² = 0.533 (best for arousal: 0.567)
- **Ridge**: R² = 0.497

### Prediction Examples
```
Song 1269: Valence=4.50, Arousal=4.17 → Scaled: -0.13, -0.21 (negative, low energy)
Song 674:  Valence=5.04, Arousal=5.96 → Scaled: 0.01, 0.24  (neutral, high energy)
Song 723:  Valence=5.64, Arousal=5.81 → Scaled: 0.16, 0.20  (positive, high energy)
```

## ✅ Current Solution

The prediction script now handles this correctly by:

1. **Accepting [1,9] predictions** from the trained models
2. **Converting to [-1,1]** for interpretation: `scaled = (original - 5.0) / 4.0`
3. **Providing interpretations** based on scaled values

## 🔧 Recommended Fixes (For Future Training)

### Option 1: Fix the DataLoader (Recommended)
```python
def _scale_annotations(self, annotations):
    """
    Scales valence and arousal from [1, 9] to [-1, 1].
    """
    logger.info("Scaling annotations from [1, 9] to [-1, 1]")
    # Look for actual column names in the data
    for col in annotations.columns:
        if 'valence' in col.lower() or 'arousal' in col.lower():
            if annotations[col].min() >= 1 and annotations[col].max() <= 9:
                annotations[col] = (annotations[col] - 5.0) / 4.0
                logger.info(f"Scaled column: {col}")
    return annotations
```

### Option 2: Retrain with Correct Scaling
1. Fix the `_scale_annotations()` function
2. Retrain all models
3. Update evaluation metrics
4. Compare performance on [-1,1] vs [1,9] scale

## 📝 Documentation Updates Needed

1. **Technical Report**: Update to reflect that models were trained on [1,9] scale
2. **Performance Metrics**: Add clarification about scale used
3. **Prediction Scripts**: Document the scale conversion process

## 🎯 Key Takeaways

1. **Models work correctly** - they were just trained on a different scale than intended
2. **Performance metrics are valid** - R² scores are still meaningful
3. **Scaling conversion works** - predictions can be interpreted on [-1,1] scale
4. **Important lesson**: Always validate preprocessing steps with sample data

## 🚀 Action Items

- [ ] Update technical documentation to reflect actual training scale
- [ ] Consider retraining with corrected scaling for consistency
- [ ] Add validation checks to preprocessing pipeline
- [ ] Update prediction scripts with scale conversion

---

**Status:** ✅ Issue identified and worked around  
**Impact:** 🟡 Medium - models work but on different scale than intended  
**Priority:** 🟡 Medium - consider fixing in next training iteration
