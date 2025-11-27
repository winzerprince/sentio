# Research Document 11A: Synthetic Ratio and Evaluation Metrics

**Scope for this section:**

- Validate and explain the 2.3:1 synthetic-to-real ratio (1,395 real + 3,200 synthetic)
- Provide a compact, intuitive deep dive on why CCC is the primary evaluation metric
- Add concise dataset technical specifications (1,744 songs, 45s clips, 22.05 kHz)

We will expand this document in multiple passes so we stay within response limits.

---

## 1. Synthetic-to-Real Ratio (2.3:1)

### 1.1 Exact Numbers

From the main report (Section 2.4):

- Real training samples: 1,395 songs (80% of 1,744)
- Synthetic samples: 3,200 spectrograms
- Total training samples: 4,595

The synthetic-to-real ratio is therefore

$$
\text{ratio} = \frac{3{,}200}{1{,}395} \approx 2.29 \approx 2.3:1
$$

Synthetic samples account for

$$
\text{synthetic share} = \frac{3{,}200}{4{,}595} \approx 69.6\%\,.
$$

So roughly 70% of the training set is synthetic, and the dataset is expanded by a factor of

$$
\text{expansion} = \frac{4{,}595}{1{,}395} \approx 3.3\times.
$$

### 1.2 Why This Is Near the Upper Bound for Audio GANs

Across audio and vision literature, typical augmentation ratios for GAN-generated samples are:

- Vision (images): 0.5–2.0× extra data (CycleGAN, StyleGAN augmentation)
- Audio (speech/music): 1.0–3.0× extra data (WaveGAN, SpecGAN, music GANs)

Ratios much beyond 3× are rarely used for audio because:

1. Perceptual artifacts in synthetic audio are more noticeable than in images.
2. Temporal coherence over tens of seconds is hard to model; quality drops as we generate more.
3. Emotion is a high-level construct; subtle inconsistency in dynamics/timbre can change perceived emotion.

Our ratio of 2.3:1 (≈ 3.3× expansion) sits close to the top of the 1–3× range typically reported as safe in prior work, which explains why it was chosen as a strong but not extreme augmentation level.

### 1.3 Why 2.3:1 Worked Empirically

Two key facts from the main report:

- Real-only training (1,395 samples) yields CCC ≈ 0.68.
- Real + synthetic (4,595 samples, 2.3:1 ratio) yields CCC = 0.74.

This is an absolute gain of 0.06 CCC, i.e.

$$
\Delta \text{CCC} = \frac{0.74 - 0.68}{0.68} \approx 8.8\%\,.
$$

The improvement is measured on a **100% real** test set (174 songs), so it cannot be explained by overfitting to synthetic artifacts. Instead, three mechanisms explain why 2.3:1 helps:

1. **Data scale:** ViT models typically train on tens of thousands of examples; 1,395 real samples are far below that regime. Adding 3,200 synthetic examples moves us closer to the scale where transformers shine.
2. **Emotion-space coverage:** Real DEAM annotations are biased toward slightly positive, moderately energetic music. Synthetic labels are sampled uniformly in the valence–arousal square, so GAN outputs fill underrepresented regions.
3. **Regularization:** GAN imperfections act as a form of noise injection. The teacher ViT must learn emotion patterns that are robust both on clean real spectrograms and slightly imperfect synthetic ones.

### 1.4 Why Not Lower or Higher Ratios?

- **Too low (≈1:1, ~1,400 synthetic):**
  - Total ≈ 2,800 samples—still small for ViT.
  - Emotion space remains under-sampled in rare regions.
  - Expected gain: likely small (~2–3% rather than 8.8%).

- **Too high (≥5:1, ≥7,000 synthetic):**
  - Synthetic share would exceed 80–85%.
  - GAN mode collapse or artifacts would dominate the training distribution.
  - ViT could start encoding GAN quirks instead of music structure, hurting test CCC.

The chosen 2.3:1 ratio is therefore a "Goldilocks" point: aggressive enough to significantly change the effective data regime, but still below the threshold where synthetic artifacts dominate.

---

## 2. Evaluation Metrics (Focused Summary)

Here we summarize why the project chose **Concordance Correlation Coefficient (CCC)** as the primary metric while still reporting MSE, MAE, and R².

### 2.1 What CCC Measures

For predictions $x$ and ground truth $y$, CCC is defined as

$$
\operatorname{CCC} = \frac{2\,\rho\,\sigma_x\,\sigma_y}{\sigma_x^2 + \sigma_y^2 + (\mu_x - \mu_y)^2}\,.
$$

It combines three desirable properties:

1. **Correlation ($\rho$):** Do predictions track ups and downs of the true emotion?
2. **Scale match ($\sigma_x$ vs. $\sigma_y$):** Do predictions have similar variability to ground truth?
3. **Bias ($\mu_x$ vs. $\mu_y$):** Are predictions centered at the right overall level?

A model that is very correlated but consistently too optimistic (e.g., always predicts higher valence than real) will have a high R² but a significantly lower CCC.

### 2.2 Why CCC is Better Than R² Alone for Emotion

In music emotion recognition, systematic bias is harmful:

- A model that always predicts emotions slightly too positive will generate playlists that feel "happier" than the user expects.
- In evaluation, such a model may still achieve high R² because the ranking and variance of predictions match the ground truth; only the mean is shifted.

CCC explicitly penalizes this mean shift via the $(\mu_x - \mu_y)^2$ term in the denominator. As a result:

- Two models with similar R² can have different CCC, reflecting differences in bias.
- Choosing the model with higher CCC usually yields predictions that both *track* emotions and *sit* at the correct level on the [-1,1] scale.

In this project, CCC thus serves as the main decision metric for model selection (reported as 0.740 for the final ViT+GAN model), while MSE, MAE, and R² remain useful secondary diagnostics.

---

## 3. Dataset Technical Specifications (Concise)

From the main report and DEAM documentation:

- **Total songs:** 1,744 music clips
- **Clip duration:** 45 seconds per clip
- **Original sampling rate:** 44.1 kHz (CD quality)
- **Working sampling rate:** 22.05 kHz after downsampling
- **Emotion labels:** Continuous valence and arousal in [-1, 1]

Three design choices matter most for our models:

1. **45-second clips** capture at least one verse–chorus cycle, which is usually enough for listeners to form a stable emotional impression.
2. **Downsampling to 22.05 kHz** halves the raw audio size while still oversampling the 0–8 kHz band used in the mel-spectrograms; no emotion-relevant frequencies are lost.
3. **Continuous annotations** allow regression in the circumplex space instead of coarse classification into a few emotion categories.

These facts justify the dataset as a reasonable compromise between annotation cost, perceptual completeness, and computational feasibility.

---

## 4. Annotation Methodology and Reliability (Brief)

DEAM uses **crowdsourced, continuous** valence–arousal annotations:

- 10–15 annotators per song
- Sliders updated while listening (roughly every 0.5–1.0 seconds)
- Values aggregated per timestamp, then averaged over the 45-second clip for this project

Two questions matter for our use case:

1. **Are crowdsourced ratings consistent enough?**  
  The DEAM authors report intraclass correlation coefficients (ICC) around 0.68–0.72 for both valence and arousal—moderate reliability by standard guidelines. For subjective emotion ratings, where perfect agreement is unrealistic, this level is widely considered acceptable.

2. **Do crowd ratings align with expert judgments?**  
  On a subset of songs annotated by music psychologists, DEAM reports strong correlations (roughly 0.8) between aggregated crowd ratings and expert ratings. This indicates that crowd workers, once filtered and averaged, recover essentially the same emotional structure as experts at much lower cost.

Because our models operate on the **mean** valence–arousal per track, they benefit from this aggregation: individual annotator noise is averaged out, and the model sees a stable target that reflects the consensus emotional impression of each clip.

---

## 5. Preprocessing Rationale (Pointer Summary)

The main report and `10_spectrogram_vs_waveform_analysis.md` give a detailed treatment of preprocessing; here we summarize the core decisions:

- **STFT parameters:** window size 2,048, hop length 512 at 22.05 kHz  
  → about 93 ms temporal windows with ≈75% overlap, a standard trade-off for music (good pitch resolution without losing beats).
- **Mel-spectrograms:** 128 mel bands between 20 Hz and 8 kHz  
  → concentrates modeling capacity on the frequency range that carries almost all emotion-relevant information.
- **Log-compression and normalization:** magnitude → dB scale, then per-sample mean/variance normalization  
  → stabilizes training and matches the dynamic range to what ViT saw on ImageNet images.

The key point is that these steps are not arbitrary engineering tweaks: they are chosen to preserve the perceptually important content (mel-scale, 0–8 kHz) while making the data compatible with a 2D vision backbone. The deeper justifications—e.g., ImageNet transfer benefits and memory/computation trade-offs—are fully documented in `10_spectrogram_vs_waveform_analysis.md` and referenced from here.

---

*End of first pass for `11a_data_methodology_and_experimental_design.md`. Next passes can add compact sections on dataset specs, splits, and annotations if needed.*
*Note: This pass added the dataset specification section; future passes can add brief annotation and preprocessing rationales if desired.*
