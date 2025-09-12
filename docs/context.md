# Harmonia - AI-Powered Emotional Music Analysis and Generation Engine

## Project Overview

**Harmonia** is an ambitious AI project that addresses a fundamental gap in current music technology. Instead of relying on surface-level attributes like genre or artist, Harmonia develops AI models capable of analyzing and generating music based on emotional parameters, ultimately creating a system that understands and replicates the emotional essence of music.

## Problem Statement

Listeners often struggle to discover music that truly resonates with them because existing recommendation systems rely heavily on surface-level attributes such as genre, artist, or album. These categories fail to capture the emotional impact that defines a listener's connection to a song. A listener may appreciate a particular section of a track for the emotions it evokes, yet find that other songs within the same genre or album fall short of producing the same effect.

This disconnect highlights the inadequacy of traditional music classification systems in addressing the core factor driving musical preference: **the emotional experience**.

## Project Objectives

### Primary Goals

1. **Analyze existing music** to map it onto a defined emotional spectrum
2. **Generate new music** that evokes specific, user-specified emotions
3. **Progress to handling and generating music** for more nuanced and obscure emotions
4. **Create a bridge** between abstract emotional concepts and sonic characteristics

### Core Innovation

Moving beyond traditional music recommendations to create a system that understands and replicates the emotional essence of music using the **Circumplex Model (Valence-Arousal)** as our primary emotional framework.

## Technical Framework

### Emotional Model: The Circumplex Approach

We utilize the **Valence-Arousal (VA) model** based on Thayer's and Russell's dimensional emotion theory:

- **Valence (V)**: The pleasantness of an emotion, ranging from negative (sad, angry) to positive (happy, serene). Represented as a value from -1 to 1.
- **Arousal (A)**: The intensity or energy level of an emotion, ranging from low (calm, sleepy) to high (excited, frantic). Represented as a value from -1 to 1.

#### Emotional Quadrants

- **High Valence, High Arousal**: Joy, excitement
- **High Valence, Low Arousal**: Serenity, contentment
- **Low Valence, High Arousal**: Anger, fear
- **Low Valence, Low Arousal**: Sadness, depression

### BRECVEMA Integration

We focus on **Calmness (C)** and **Activation (A)** from the BRECVEMA framework:

- **C (Calmness)**: Maps to low arousal in our circumplex model
- **A (Activation)**: Maps to high arousal in our circumplex model

## Four-Phase Development Plan

### Phase 1: Emotional Analysis and Mapping

**Objective**: Develop a model that can analyze musical tracks and map them onto a quantitative emotional spectrum.

**Technical Approach**:

- **Data**: DEAM, PMEmo datasets with VA annotations
- **Features**: Mel spectrograms, MFCCs, spectral features, chroma, tempo
- **Model**: Convolutional Recurrent Neural Network (CRNN)
- **Output**: Continuous (V,A) coordinates creating an "emotional arc" of songs

**Expected Outcome**: A trained model that outputs temporal emotional coordinates for any audio input.

### Phase 2: Music Generation from Core Emotions

**Objective**: Generate novel musical clips based on user-specified emotional coordinates.

**Technical Approach**:

- **Architecture**: Conditional Variational Autoencoder (VAE) or Transformer-based models
- **Input**: Emotional (V,A) vectors
- **Training**: Same dataset from Phase 1, learning to reverse the analysis process
- **Output**: Generated audio segments matching target emotions

**Expected Outcome**: Interactive system where users select emotional coordinates and receive generated music.

### Phase 3: Interpreting Obscure and Nuanced Emotions

**Objective**: Bridge the gap between abstract emotional concepts and sonic characteristics.

**Technical Approach**:

- **Semantic Embedding**: Large Language Models to convert emotional definitions to vectors
- **Cross-Modal Learning**: Dual-encoder model with contrastive learning (CLIP-style)
- **Data**: Songs with rich textual descriptions, reviews, tags
- **Capability**: Zero-shot emotion analysis and music retrieval

**Expected Outcome**: System that can analyze music for obscure emotions and retrieve music by emotional descriptions.

### Phase 4: Generating Music for Obscure Emotions

**Objective**: Synthesize novel music designed to evoke specific, nuanced emotions from textual prompts.

**Technical Approach**:

- **Integration**: Combine Phase 2 generation with Phase 3 semantic understanding
- **Input**: Textual descriptions of emotions (e.g., "liberosis: the desire to care less")
- **Process**: Text → Semantic embedding → Conditioned music generation
- **Output**: Novel compositions designed to evoke specified emotions

**Expected Outcome**: Complete system for emotional music creation from natural language descriptions.

## Datasets and Resources

### Primary Datasets

#### 1. DEAM (Database for Emotion Analysis in Music)

- **Description**: 1,802 music clips with continuous VA annotations
- **Features**: Moment-by-moment emotional ratings
- **Access**: [DEAM Dataset Page](https://www.tensorflow.org/datasets/catalog/deam)
- **Use Case**: Primary training data for Phase 1 & 2

#### 2. PMEmo (1000 Song Corpus for Emotion Recognition)

- **Description**: 1000 full-length songs with static and dynamic VA annotations
- **Features**: Both overall and temporal emotional ratings
- **Access**: [PMEmo on GitHub](https://github.com/HuiZhangDB/PMEmo2019)
- **Use Case**: Validation and additional training data

#### 3. GEMS (Geneva Emotional Music Scales)

- **Description**: 300 instrumental excerpts with detailed emotional annotations
- **Features**: Multiple emotional scales including VA
- **Access**: [GEMS Dataset Page](https://www.unige.ch/cisa/research/materials-and-equipments/research-material/geneva-emotional-music-scale-gems/)
- **Use Case**: Cross-validation and model testing

#### 4. Spotify Web API

- **Description**: Real-time access to audio features including valence and energy
- **Features**: 13 audio features per track, including valence and energy proxies
- **Access**: [Spotify API Documentation](https://developer.spotify.com/documentation/web-api/reference/get-audio-features)
- **Use Case**: Large-scale dataset creation and real-world validation

### Conceptual Resources

- **The Dictionary of Obscure Sorrows**: John Koenig's collection of invented words for emotional states
- **Circumplex Model Literature**: Russell (1980), Thayer (1989) foundational papers

## Technical Toolchain (Arch Linux)

### System Dependencies

```bash
# Core system packages
sudo pacman -Syu ffmpeg python-pip git

# Optional: GPU support
sudo pacman -S cuda cudnn  # For NVIDIA GPUs
```

### Python Environment Setup

```bash
# Create virtual environment
python -m venv harmonia_env
source harmonia_env/bin/activate

# Core ML and audio libraries
pip install torch torchvision torchaudio  # or tensorflow
pip install librosa pandas numpy matplotlib seaborn
pip install scikit-learn jupyterlab soundfile
pip install transformers datasets accelerate
pip install pydub scipy
```

### Audio Processing Libraries

#### Primary: Librosa

- **Purpose**: Comprehensive audio analysis and feature extraction
- **Key Features**: MFCC, spectrograms, tempo, chroma, tonnetz
- **Documentation**: [Librosa Docs](https://librosa.org/doc/latest/index.html)

#### Supporting Libraries

- **FFmpeg**: Audio format conversion and preprocessing
- **SoundFile**: Audio I/O operations
- **PyDub**: Audio manipulation and effects
- **Essentia**: Advanced music analysis (optional)

### Deep Learning Frameworks

#### PyTorch (Recommended)

- **Reason**: Better flexibility for research, strong audio support
- **Audio Extensions**: torchaudio, asteroid-filterbanks
- **Installation**: `pip install torch torchaudio`

#### TensorFlow (Alternative)

- **Reason**: Excellent for deployment, comprehensive ecosystem
- **Audio Extensions**: tensorflow-io, kapre
- **Installation**: `pip install tensorflow tensorflow-io`

### Development Workflow

#### 1. Data Preprocessing Pipeline

```bash
# Example preprocessing workflow
ffmpeg -i input.mp3 -ac 1 -ar 22050 -sample_fmt s16 output.wav  # Standardize
python scripts/extract_features.py --input data/raw --output data/processed
python scripts/clean_labels.py --annotations annotations.csv --output clean_labels.csv
```

#### 2. Feature Extraction Strategy

- **Target Format**: WAV, 16-bit, 22.05kHz, mono
- **Segment Length**: 30 seconds for consistency
- **Feature Set**:
  - Rhythm: Tempo, beat density
  - Timbre: MFCCs (20 coefficients), spectral features
  - Harmony: Chroma features (12), tonnetz (6)
  - Energy: RMS energy, spectral rolloff

#### 3. Data Cleaning Approach

- **Audio Normalization**: Peak normalization to -0.1 dB
- **Silence Detection**: Remove leading/trailing silence using librosa
- **Quality Control**: Remove corrupted or silent files
- **Label Validation**: Check for NaN values, outliers outside [-1, 1] range

### Development Tools

#### Code Quality

```bash
pip install black flake8 mypy  # Code formatting and linting
pip install pytest pytest-cov  # Testing framework
```

#### Experiment Tracking

```bash
pip install wandb mlflow  # Experiment logging
pip install tensorboard  # Visualization
```

#### Jupyter Environment

```bash
pip install jupyterlab ipywidgets
jupyter lab  # Interactive development
```

### Project-Specific Scripts

#### Feature Extraction Script Structure

```python
# src/data_processing/audio_features.py
import librosa
import numpy as np
import pandas as pd

def extract_comprehensive_features(audio_path, sr=22050, duration=30):
    """Extract all relevant audio features for emotion analysis"""
    # Implementation details in actual codebase
    pass

def batch_process_dataset(dataset_path, output_path):
    """Process entire dataset and save features"""
    # Implementation details in actual codebase
    pass
```

## Data Management Strategy

### Directory Structure

```
data/
├── raw/                    # Original, unprocessed datasets
│   ├── DEAM/              # DEAM dataset files
│   ├── PMEmo/             # PMEmo dataset files
│   └── spotify/           # Spotify API downloads
├── processed/             # Cleaned, standardized data
│   ├── features/          # Extracted audio features
│   ├── annotations/       # Cleaned emotional labels
│   └── splits/            # Train/validation/test splits
└── external/              # External datasets and APIs
```

### Version Control Strategy

- **Git LFS**: For large audio files and datasets
- **DVC**: For data versioning and pipeline management
- **Exclude**: Raw audio files, model checkpoints, large datasets

## Evaluation Metrics

### Phase 1 (Analysis)

- **Regression Metrics**: MSE, MAE for VA prediction
- **Correlation**: Pearson correlation with human annotations
- **Temporal Consistency**: Smoothness of emotional trajectories

### Phase 2 (Generation)

- **Perceptual Quality**: Human evaluation studies
- **Emotional Accuracy**: Generated music's measured emotions vs. targets
- **Diversity**: Variety in generated outputs for same emotional input

### Phase 3 & 4 (Semantic)

- **Semantic Similarity**: Cosine similarity in embedding space
- **Human Evaluation**: Subjective assessment of emotional matching
- **Cross-Modal Retrieval**: Precision@K for text-to-music search

## Expected Challenges and Solutions

### Technical Challenges

1. **Data Scarcity**: Limited emotionally-annotated music datasets
   - **Solution**: Data augmentation, transfer learning, synthetic data
2. **Computational Requirements**: Large-scale audio processing
   - **Solution**: Efficient feature extraction, distributed computing
3. **Subjective Evaluation**: Emotional response is personal
   - **Solution**: Multiple annotators, demographic considerations

### Research Challenges

1. **Emotion Complexity**: Mapping abstract concepts to audio
   - **Solution**: Multi-modal learning, semantic embeddings
2. **Generalization**: Models working across music styles
   - **Solution**: Diverse training data, domain adaptation
3. **Real-time Performance**: Interactive applications
   - **Solution**: Model optimization, efficient architectures

## Success Criteria

### Short-term (Phases 1-2)

- [ ] Achieve correlation >0.7 with human VA annotations
- [ ] Generate recognizable emotional content (user studies >70% accuracy)
- [ ] Real-time feature extraction (<1s for 30s audio)

### Long-term (Phases 3-4)

- [ ] Handle 100+ obscure emotional concepts
- [ ] Generate emotionally coherent music from text descriptions
- [ ] Deploy interactive web application for public use

## Future Directions

### Technical Extensions

- **Multi-modal Integration**: Combine audio with lyrics, album art
- **Personalization**: User-specific emotional mappings
- **Cultural Considerations**: Cross-cultural emotional responses

### Applications

- **Therapeutic Music**: Emotion-targeted music therapy
- **Content Creation**: Soundtrack generation for media
- **Music Education**: Understanding emotional expression in music
- **Accessibility**: Emotional audio descriptions for visually impaired

## References and Further Reading

### Foundational Papers

- Russell, J. A. (1980). A circumplex model of affect. _Journal of Personality and Social Psychology_, 39(6), 1161-1178.
- Thayer, R. E. (1989). _The Biopsychology of Mood and Arousal_. Oxford University Press.

### Music Emotion Recognition

- Yang, Y. H., & Chen, H. H. (2012). Music emotion recognition: A survey. _Information Retrieval_, 15(5), 409-434.
- Delbouys, R., et al. (2018). Music mood detection based on audio lyrics with deep neural networks. _Neurocomputing_, 275, 1213-1230.

### Music Generation

- Briot, J. P., Hadjeres, G., & Pachet, F. D. (2017). _Deep Learning Techniques for Music Generation_. Springer.
- Dhariwal, P., et al. (2020). Jukebox: A generative model for music. _arXiv preprint arXiv:2005.00341_.

---

_Last Updated: September 4, 2025_
_Project Status: Initial Setup and EDA Phase_
