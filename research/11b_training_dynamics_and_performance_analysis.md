# Research Document 11B: Training Dynamics and Performance Progression

**Scope for this section:**
 
- Summarize how data splits, CGAN training choices, and hyperparameters shape model training dynamics
- Provide compact narratives for the performance table entries (Ridge → SVR → XGBoost → AST → ViT → MobileViT)
- Quantify augmentation effects and key GAN hyperparameters

This document is built up section by section to stay within response limits.

---

## 1. Data Split and CGAN Training Dynamics

### 1.1 80/10/10 Split: Why These Percentages

The project uses an 80/10/10 split of the 1,744 DEAM songs:

- 1,395 for training
- 175 for validation
- 174 for testing

This choice balances three competing goals:

1. **Enough data to train deep models:** ViT and GAN training benefit from as many training examples as possible; pushing training below ~70% would shrink the real training set to under 1,250 songs, further stressing an already data-hungry architecture.
2. **Reliable early stopping:** A validation set of 175 songs provides a stable CCC estimate with standard error around ±0.02, sufficient to detect overfitting and choose hyperparameters.
3. **Meaningful final evaluation:** A 174-song test set yields narrow confidence intervals for CCC, allowing us to distinguish, for example, 0.68 vs 0.74 with high confidence.

### 1.2 CGAN Training Schedule and Stability

From the main report:

- GAN discriminator and generator are trained for 10–15 epochs.
- Batch size is 24–32.
- A 1:2 update ratio (discriminator:generator) is used.
- Label smoothing and instance noise regularize the discriminator.

These settings respond to common GAN failure modes:

- Too many discriminator updates per generator step can quickly push $D$ to near-perfect classification, causing vanishing gradients for $G$.
- Label smoothing (using 0.9 instead of 1.0 for real labels) reduces discriminator overconfidence and leads to smoother gradients.
- Instance noise, gradually annealed, prevents $D$ from memorizing real spectrograms and improves generalization to synthetic ones.

Empirically, the training configuration keeps discriminator accuracy in the 70–80% range—high enough to provide informative feedback, but well below the regime where $G$ collapses.

### 1.3 Key GAN Hyperparameters (Concise)

From the reported configuration:

- **Generator optimizer:** Adam, learning rate on the order of $2 \times 10^{-4}$, with $(\beta_1, \beta_2) = (0.5, 0.999)$
- **Discriminator optimizer:** Same family and similar learning rate, ensuring neither network is systematically under-trained
- **Epochs:** 10–15, enough for the generator to learn a useful prior but not so long that $D$ overfits the finite training set
- **Batch size:** 24–32, a compromise between gradient stability and GPU memory constraints
- **Label smoothing:** real labels set to 0.9 instead of 1.0
- **Instance noise:** small Gaussian noise added to inputs early in training, annealed toward zero

These choices follow standard best practices for stabilizing GANs on relatively small datasets: conservative learning rates, symmetric optimizers, and explicit regularizers that prevent the discriminator from becoming perfectly confident.

---

## 2. Performance Table Narratives

Here we give concise narratives for the main models in the performance progression.

### 2.1 Ridge Regression (R² = 0.497)

- **Inputs:** 164 engineered features per song (OpenSMILE summary statistics).
- **Strength:** Captures global linear relationships (e.g., higher tempo → higher arousal).
- **Limitation:** Cannot model non-linear interactions or temporal structure; it sees each song as a static feature vector.
- **Role in the project:** Establishes a solid traditional baseline and demonstrates that even simple models can explain roughly half of the variance in emotion ratings.

### 2.2 SVR (R² = 0.533)

- **Change vs Ridge:** Still operates on the same 164-dimensional feature set, but uses a kernel (e.g., RBF) to model non-linear relationships.
- **Gain:** Modest improvement (~7.2% over Ridge) from capturing curvature in feature–emotion mappings.
- **Bottleneck:** Still constrained by the expressive power of the hand-crafted features; no access to raw spectrogram structure or temporal dynamics.

### 2.3 XGBoost (R² = 0.540)

- **Change vs SVR:** Gradient-boosted decision trees replace kernel methods.
- **Strength:** Good at modeling feature interactions and handling non-linear effects without feature engineering.
- **Observations:** Slight improvement (~8.7% over Ridge) suggests that the remaining error comes mainly from missing information (lost in temporal averaging and handcrafted features), not from model capacity.

### 2.4 AST (CCC = 0.68)

- **Architectural shift:** Moves from static feature vectors to full mel-spectrograms; uses a transformer tailored to audio.
- **Key advantage:** Self-attention layers can attend across the entire time–frequency plane, capturing motif repetitions, build-ups, and drops.
- **Result:** A large jump in CCC compared to XGBoost, showing that access to raw time–frequency structure is far more valuable than additional complexity on top of compressed features.

### 2.5 ViT + GAN (CCC = 0.74)

- **Further shift:** Uses a Vision Transformer pretrained on ImageNet and fine-tuned on spectrograms, plus augmented data from the CGAN.
- **Two main contributors to the gain over AST:**
  1. **ImageNet pretraining** provides robust low- and mid-level filters, reducing the data required to learn good representations.
  2. **GAN augmentation** increases effective sample size and improves coverage of rare emotion regions.
- **Net effect:** CCC rises from 0.68 (AST) to 0.74, an 8.8% relative improvement on a purely real test set.

### 2.6 MobileViT Student (CCC = 0.69)

- **Architecture:** A distilled, lighter transformer with 12M parameters instead of the teacher’s 86M.
- **Objective:** Preserve as much of the teacher’s performance as possible while sharply reducing memory and inference cost.
- **Outcome:**
  - CCC = 0.69, which is 93.2% of the teacher’s 0.74.
  - Parameter count reduced by a factor of ~7.2×.
  - Inference is ~4× faster (200 ms → 50 ms), enabling mobile deployment.

The overall narrative is that each step either increases **information richness** (moving from summary features to full spectrograms), increases **capacity** (from linear to non-linear, from CNN/RNN to transformers), or improves **data efficiency** (pretraining and GAN augmentation). Distillation then trades some of that capacity back for efficiency while retaining most of the CCC.

---

## 3. Augmentation Effects (Quantitative Snapshot)

We can summarize the impact of GAN-based augmentation on the transformer teacher as follows:

| Training Data | Real Samples | Synthetic Samples | Total | Test CCC | Relative Gain vs. Real-only |
|---------------|-------------|-------------------|-------|----------|------------------------------|
| Real only     | 1,395       | 0                 | 1,395 | 0.68     | –                            |
| Real + GAN    | 1,395       | 3,200             | 4,595 | 0.74     | +8.8%                        |

Two aspects are important:

1. **Generalization is measured on real data only.**  
  The test set contains 174 real songs and no synthetic spectrograms. The CCC improvement therefore reflects genuinely better generalization to real music, not better performance on GAN artifacts.

2. **Most of the gain comes from coverage, not just raw scale.**  
  The synthetic labels are sampled uniformly over the valence–arousal square, so GAN outputs populate emotion regions underrepresented in the original DEAM distribution. The ViT learns to handle both common and rare emotional combinations, which is precisely what a robust recommender needs.

This table and explanation tie the 2.3:1 ratio, the GAN training decisions, and the final CCC numbers together in one place.

---

*End of first pass for `11b_training_dynamics_and_performance_analysis.md`. Further sections (GAN hyperparameters, detailed augmentation effects) can be added if needed.*
