# Quick Fix Reference - Distilled ViT Notebook

## 🔧 All Issues Fixed

### ✅ NameError Fixed
Changed `vit_model` → `model` in cells 24, 26, 27

### ✅ GAN Training Balanced
- Discriminator LR: 50% of generator LR
- Train discriminator: every 2 iterations
- Train generator: 2x per discriminator update
- Label smoothing: real=0.9, fake=0.1
- Instance noise with decay
- Gradient clipping

### ✅ ViT Training Improved
- Deeper regression head (3 layers)
- Partial fine-tuning (last 4 blocks)
- Differential learning rates (backbone: 10%, head: 100%)
- Gradient accumulation (4 steps)
- Save best model by CCC (not loss)
- More epochs (40 instead of 30)

### ✅ Better Hyperparameters
```python
GAN_EPOCHS = 15    # Was 10
VIT_EPOCHS = 40    # Was 30
VIT_LR = 3e-5      # Was 1e-4
```

## 🚀 Expected Results

### GAN Monitoring
```
Epoch 5/15 | D_loss: 0.520 | G_loss: 0.680 | D_real: 0.85 | D_fake: 0.35
                   ↑ Good!   ↑ Good!         ↑ Good!       ↑ Improving!
```

### ViT Training
```
Epoch 20/40 | CCC_V: 0.68 | CCC_A: 0.71 | Avg: 0.695
                    ↑ Target > 0.65 for good performance
```

### Distillation
```
Student CCC: 0.625 | Teacher CCC: 0.695 | Retention: 89.9%
                                              ↑ Target > 85%
```

## 📊 Success Criteria

| Metric | Target | Good | Excellent |
|--------|--------|------|-----------|
| GAN D_loss | 0.3-0.8 | 0.4-0.7 | 0.45-0.65 |
| GAN G_loss | 0.3-0.8 | 0.4-0.7 | 0.45-0.65 |
| ViT CCC | > 0.50 | > 0.65 | > 0.75 |
| Distillation | > 80% | > 85% | > 90% |

## 🏃 Run Order

1. **Cells 1-3**: Setup ✅
2. **Cells 4-5**: Load DEAM ✅
3. **Cells 6-9**: GAN Training (monitor D_real/D_fake)
4. **Cell 10-11**: Generate synthetic data
5. **Cells 12-13**: Prepare datasets
6. **Cells 14-19**: ViT Training (monitor CCC)
7. **Cells 20-22**: Evaluate ViT
8. **Cells 23-27**: Knowledge Distillation

## 🆘 If Issues Persist

### GAN not learning?
```python
# In cell 3, try:
GAN_EPOCHS = 20
GAN_LR = 0.0001  # Even lower
```

### ViT CCC still low?
```python
# In cell 3, try:
VIT_EPOCHS = 60
VIT_LR = 1e-5  # Even lower
```

### Distillation poor?
```python
# In cell 26, try:
DISTILL_EPOCHS = 20
distill_optimizer = torch.optim.AdamW(..., lr=1e-4)  # Lower LR
```

---

**Ready to execute!** Run cells sequentially and monitor the metrics above.
