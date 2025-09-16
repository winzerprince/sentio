# AI Music Emotion Analysis & Generation - Comprehensive Testing Plan

This repository contains a comprehensive testing framework for evaluating different AI models for music emotion analysis and generation, specifically designed for the HP EliteBook 840 G3 system constraints.

## 🎯 Project Overview

The goal is to systematically test and compare various AI models for:
1. **Emotion Analysis**: Predicting valence and arousal from music features
2. **Music Generation**: Creating new music based on emotional parameters

## 🔧 System Specifications
- **Computer**: HP EliteBook 840 G3
- **CPU**: Intel i5-6300U (2.4-3.0GHz, 2 cores, 4 threads)
- **RAM**: 16GB
- **GPU**: Intel HD Graphics 520 (integrated)
- **OS**: Linux

## 📊 Dataset: DEAM (Database for Emotion Analysis using Music)
- **Total Songs**: 1,802 with pre-extracted features
- **Features per Song**: 260+ audio features (MFCCs, spectral features, etc.)
- **Emotion Labels**: Valence & Arousal on 1-9 scale (normalized to 0-1)
- **Training Sample**: 1,200 songs (optimized for system constraints)

## 🧠 Models Being Tested

### Emotion Analysis Models

#### Lightweight Models (Fast, Low Memory)
1. **Ridge Regression**
   - Linear model with regularization
   - ✅ Very fast training (seconds)
   - ✅ Interpretable coefficients
   - ❌ Only captures linear relationships

2. **Support Vector Regression (SVR)**
   - Non-linear regression with RBF kernel
   - ✅ Captures non-linear patterns
   - ✅ Memory efficient
   - ❌ Slower than Ridge

#### Heavyweight Models (Higher Accuracy, More Resources)
3. **XGBoost**
   - Gradient boosting with decision trees
   - ✅ Often best performance on tabular data
   - ✅ Feature importance analysis
   - ❌ Computationally intensive

4. **Multi-layer Perceptron (MLP)**
   - Neural network with hidden layers
   - ✅ Highly flexible, can model complex patterns
   - ✅ Single model for both valence & arousal
   - ❌ Requires feature scaling, data hungry

### Music Generation Models

#### Lightweight Model
1. **Markov Chains**
   - Probabilistic state transitions
   - ✅ Simple to understand and implement
   - ✅ Fast training on CPU
   - ❌ Limited creativity, lacks long-term structure

#### Heavyweight Model
2. **Simplified Conditional VAE (CVAE)**
   - Variational autoencoder with emotion conditioning
   - ✅ Can generate novel music
   - ✅ Fine-grained emotional control
   - ❌ Complex, memory intensive

## 📋 Evaluation Metrics

### Analysis Models
- **RMSE (Root Mean Square Error)**: Lower is better
- **MAE (Mean Absolute Error)**: Lower is better  
- **R² Score**: Higher is better (max = 1.0)
- **Training Time**: Minutes
- **Memory Usage**: Peak GB during training

### Generation Models
- **Training Success**: Did the model train without errors?
- **Generation Capability**: Can it generate for different emotions?
- **Training Time**: Seconds/Minutes
- **Memory Usage**: Peak GB during training
- **Sample Quality**: Subjective assessment

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Complete Testing Suite
```bash
python run_comprehensive_tests.py
```

This will:
- Test all analysis models with cross-validation
- Test generation models (if memory allows)
- Generate comprehensive reports and visualizations
- Monitor system resources throughout

### 3. Run Individual Tests

**Analysis Models Only:**
```bash
python src/model_testing_framework.py
```

**Generation Models Only:**
```bash
python src/generation_testing_framework.py
```

## 📁 Project Structure

```
sentio/
├── dataset/DEAM/                          # DEAM dataset
│   ├── annotations/                       # Emotion labels
│   ├── features/                          # Pre-extracted audio features
│   └── MEMD_audio/                        # Original audio files
├── src/
│   ├── model_testing_framework.py         # Analysis model testing
│   ├── generation_testing_framework.py    # Generation model testing
│   └── __init__.py
├── results/                               # Test results and reports
├── notebooks/
│   └── 1.0-data_exploration.ipynb        # Data exploration
├── run_comprehensive_tests.py             # Main execution script
├── requirements.txt                       # Python dependencies
└── README.md                              # This file
```

## 📈 Expected Results

### Analysis Model Performance (Estimated)
| Model | Valence R² | Arousal R² | Training Time | Memory Usage |
|-------|------------|------------|---------------|--------------|
| Ridge | 0.45-0.55  | 0.40-0.50  | <1 min        | <2 GB        |
| SVR   | 0.50-0.60  | 0.45-0.55  | 2-5 min       | <3 GB        |
| XGBoost | 0.60-0.70 | 0.55-0.65  | 5-15 min      | 4-8 GB       |
| MLP   | 0.55-0.65  | 0.50-0.60  | 10-30 min     | 6-12 GB      |

### System Resource Guidelines
- **Safe Memory Usage**: <12 GB (leave 4 GB for system)
- **Training Time Limits**: 
  - Lightweight models: <5 minutes
  - Heavyweight models: <30 minutes
- **CPU Usage**: Will utilize both cores efficiently

## 📊 Output Reports

After testing, you'll find in `results/`:

1. **model_comparison.csv**: Performance comparison table
2. **model_comparison.png**: Visualization charts
3. **detailed_results.json**: Complete model results and parameters
4. **generation_model_comparison.csv**: Generation model results
5. **comprehensive_test_report.json**: Final summary report

## 🔍 Understanding the Results

### Key Questions Answered:
1. **Which model gives the best accuracy for emotion prediction?**
2. **What's the trade-off between accuracy and training time?**
3. **How much memory does each model require?**
4. **Can we generate music on this hardware setup?**
5. **Which approach is most suitable for real-time applications?**

### Interpreting R² Scores:
- **0.7-1.0**: Excellent prediction
- **0.5-0.7**: Good prediction  
- **0.3-0.5**: Moderate prediction
- **<0.3**: Poor prediction

## ⚠️ System Constraints & Optimizations

### Memory Management:
- Data loaded in batches to prevent memory overflow
- Automatic garbage collection after intensive operations
- Conservative parameter grids for hyperparameter tuning
- Early stopping for neural networks

### CPU Optimization:
- Limited parallel processing (max 2 cores for training)
- Reduced dataset size (1,200 samples instead of full 1,802)
- Simplified model architectures where appropriate

### Time Constraints:
- Maximum 30 minutes per heavyweight model
- Grid search limited to essential parameters
- Early stopping implemented where possible

## 🛠️ Troubleshooting

### Common Issues:

**Memory Errors:**
- Reduce `SAMPLE_SIZE` in scripts (default: 1200)
- Close other applications before testing
- Restart if memory usage exceeds 14GB

**Long Training Times:**
- Reduce parameter grid search space
- Use smaller validation sets
- Skip heavyweight models if needed

**Audio Processing Errors:**
- Ensure MEMD_audio directory exists
- Check audio file formats (MP3 expected)
- Install additional audio codecs if needed

## 📚 Next Steps

After completing the tests:

1. **Analyze Results**: Compare model performance vs. computational cost
2. **Select Best Model**: Choose based on your accuracy/speed requirements  
3. **Fine-tune**: Optimize hyperparameters of the best performing model
4. **Deploy**: Implement the chosen model in your application

## 🤝 Contributing

Feel free to:
- Add new models to test
- Improve memory optimization
- Add more evaluation metrics
- Enhance visualization reports

## 📄 License

This project is for educational and research purposes.

---

**Happy Testing! 🎵🤖**
