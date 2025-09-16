# Sentio: AI Music Generation Based on Emotion

Sentio is an AI-powered music generation system that creates music based on emotional parameters. Unlike traditional music recommendation systems, Sentio understands and replicates the emotional essence of music to generate new compositions that evoke specific feelings.

## Project Overview

The project aims to:

1. **Analyze existing music** to map it onto a defined emotional spectrum
2. **Generate new music** that evokes specific, user-specified emotions  
3. **Handle nuanced emotions** including obscure and complex emotional states

## Features

- 🎵 **Emotional Music Classification**: Deep learning models that classify music by emotional content
- 🎼 **Conditional Music Generation**: Generate new music based on specified emotional parameters
- 🧠 **Advanced Emotion Understanding**: Support for complex and nuanced emotions
- 📊 **Comprehensive Analysis**: Audio feature extraction and emotional mapping

## Dataset

This project uses the DEAM (Database for Emotion Analysis using Music) dataset, which provides:
- Audio files with emotional annotations
- Valence and Arousal ratings
- Rich metadata for training emotional classification models

## Installation

1. Clone the repository:
```bash
git clone https://github.com/winzerprince/sentio.git
cd sentio
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
sentio/
├── data/
│   ├── raw/                    # Raw audio files and datasets
│   └── processed/              # Processed features and labels
├── src/
│   ├── data_processing/        # Audio feature extraction
│   ├── models/                 # ML models (classifier & generator)
│   ├── inference/              # Model inference scripts
│   └── utils/                  # Utility functions
├── notebooks/                  # Jupyter notebooks for exploration
├── scripts/                    # Training and execution scripts
└── requirements.txt
```

## Usage

### 1. Data Processing
Audio features are already extracted in the "selected" folder with each song's features in separate CSV files.

### 2. Train Emotion Prediction Models

#### For Static Emotion Prediction

Train models on static emotion annotations (overall ratings for each song):

```bash
bash scripts/train_static_models.sh
```

This trains three types of models:

- Linear/Ridge Regression
- Support Vector Regression (SVR) with different kernels
- XGBoost Regression

#### For Dynamic Emotion Prediction

Train models on dynamic emotion annotations (time-varying emotions):

```bash
bash scripts/train_dynamic_models.sh
```

### 3. Compare Model Performance

After training, compare the performance of different models:

```bash
python src/utils/evaluate_results.py --results_dir output/results
```

The evaluation provides metrics including R², RMSE, and MAE for each model type.

### 4. Generate Music (Future Phase)

Generate music based on emotional input (upcoming feature):

```bash
bash scripts/generate_music.sh --emotion "joy" --duration 30
```

## Development Phases

### Phase 1: Emotional Classification ✅

- [x] Audio feature extraction (MFCCs, Chroma, Mel Spectrograms)
- [x] Traditional ML models for emotion regression (Linear/Ridge, SVR, XGBoost)
- [x] Training pipeline and model evaluation

### Phase 2: Music Generation 🚧

- [ ] VAE/GAN architecture for conditional music generation
- [ ] Emotion-conditioned latent space modeling
- [ ] Audio synthesis and output generation

### Phase 3: Advanced Emotions 📋

- [ ] Extended emotional vocabulary
- [ ] Support for "Dictionary of Obscure Sorrows" emotions
- [ ] Nuanced emotional parameter handling

### Phase 4: Optimization 📋

- [ ] Model architecture improvements
- [ ] Real-time generation capabilities
- [ ] User interface development

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'feat: add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- DEAM Dataset creators for providing emotional music annotations
- The research community working on music information retrieval and generation
- Contributors to the open-source libraries that make this project possible

## Contact

- Project Maintainer: [Your Name]
- Repository: [https://github.com/winzerprince/sentio](https://github.com/winzerprince/sentio)
- Issues: [https://github.com/winzerprince/sentio/issues](https://github.com/winzerprince/sentio/issues)
