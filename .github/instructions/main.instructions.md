---
applyTo: "**"
---

## Copilot Instructions: AI Music Generation Based on Emotion

This document provides a comprehensive set of instructions for Copilot to guide the development of a project that generates music based on emotional parameters. The goal is to move beyond traditional music recommendations and create a system that understands and replicates the emotional essence of music.

### **Project Goal**

The primary objective is to build an AI model that can:

1.  Analyze existing music to map it onto a defined emotional spectrum.
2.  Generate new music that evokes specific, user-specified emotions.
3.  Progress to handling and generating music for more nuanced and obscure emotions.

---

### **Repository Structure**

Follow this folder structure to keep the project organized. Create these directories at the root of the project and populate them with the appropriate files as development progresses.

```
/
├── .gitignore
├── README.md
├── requirements.txt
├── notebooks/
│   ├── 1.0-data_exploration.ipynb
│   └── 2.0-model_prototyping.ipynb
├── data/
│   ├── raw/
│   │   ├── initial_dataset/
│   │   └── new_audio_files/
│   └── processed/
│       ├── features/
│       └── emotion_labels.csv
├── src/
│   ├── data_processing/
│   │   ├── __init__.py
│   │   └── audio_features.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── emotional_classifier.py
│   │   ├── music_generator.py
│   │   └── common.py  # Shared utilities for models
│   ├── inference/
│   │   ├── __init__.py
│   │   └── predict_emotion.py
│   ├── utils/
│   │   ├── __init__.py
│   │   └── plot_metrics.py
│   └── main.py
└── scripts/
    ├── train_classifier.sh
    └── generate_music.sh
```

---

### **Prompt Engineering Guidelines**

Use these instructions to guide your code generation and file management tasks.

#### **Initial Setup**

- **Task:** Create the foundational project structure.
- **Prompt:** "Initialize the project directory. Create the folders listed in the project structure. Generate a `.gitignore` file for a typical Python project and a basic `requirements.txt` file with common ML libraries like `torch` or `tensorflow`, `librosa`, `numpy`, and `matplotlib`."

#### **Phase 1: Emotional Classification Model**

- **Objective:** Develop a model that can classify songs into emotional categories.
- **Guidance:** This is the core of the project's first milestone.
  - **Data Processing:**
    - **Prompt:** "In `src/data_processing/`, create a module `audio_features.py`. The module should contain functions to load audio files, extract features like MFCCs, Chroma features, and Mel Spectrograms using `librosa`, and label the data from a metadata file. Include functions for data normalization and splitting into training/validation/test sets."
  - **Model Development:**
    - **Prompt:** "In `src/models/`, create a module `emotional_classifier.py`. Implement a deep learning model for audio classification. Start with a CNN-RNN hybrid architecture (e.g., a few convolutional layers followed by GRU or LSTM layers) to capture both spatial and temporal features of the audio. The model should be trained on the extracted features to predict emotional labels."
  - **Training & Evaluation:**
    - **Prompt:** "Write a script in `scripts/train_classifier.sh` to execute the training process. The script should call `src/main.py`. In `src/main.py`, create a training loop that loads the processed data, initializes the classifier model, trains it, and saves the best model checkpoint to `models/`."

#### **Phase 2: Music Generation Model**

- **Objective:** Generate music based on an emotional input.
- **Guidance:** This is a complex task. Focus on a generative approach.
  - **Model Selection:**
    - **Prompt:** "In `src/models/`, create a new module `music_generator.py`. The model should be a generative architecture, such as a Variational Autoencoder (VAE) or a Generative Adversarial Network (GAN), conditioned on emotional parameters. The VAE will encode an emotion into a latent space and decode it into a sequence of audio features. Start by implementing a simple VAE for this purpose."
  - **Inference:**
    - **Prompt:** "In `src/inference/`, create a module `generate_emotion.py`. This script should load the trained music generation model, take a user-specified emotional parameter (e.g., 'sadness', 'joy'), sample from the model's latent space, and generate a new audio sequence. The script should then save the output as a `.wav` file."

#### **Phase 3 & 4: Handling Nuanced Emotions**

- **Objective:** Progressively handle more complex and obscure emotions from "The Dictionary of Obscure Sorrows".
- **Guidance:** This phase will require significant iteration.
  - **Prompt:** "When the initial models are functional, focus on expanding the emotional label set. This may require manual labeling or a new classification approach. Adapt the data processing and model architectures to handle a much larger, more nuanced output space. Use this as an opportunity to refactor and improve the existing codebase."

---

### **General Development Principles**

- **Commit Messages:** Use a consistent format (e.g., `feat:`, `fix:`, `docs:`) to clearly describe changes.
- **Documentation:** Add docstrings to all functions and classes.
- **Dependencies:** Add any new libraries to `requirements.txt` as you introduce them.
- **Incremental Progress:** Focus on one task at a time and ensure each component works before moving to the next.
