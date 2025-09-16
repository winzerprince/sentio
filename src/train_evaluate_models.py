"""
Train, Evaluate, and Save Emotion Analysis Models

This script trains multiple regression models on the DEAM dataset,
evaluates their performance, and saves the trained models and
evaluation metrics for later analysis.
"""

import pandas as pd
import numpy as np
import os
import json
import pickle
from datetime import datetime

# Scikit-learn for modeling and evaluation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Gradient Boosting
import xgboost as xgb

# Define file paths
BASE_PATH = "dataset/DEAM"
ANNOTATIONS_PATH_1 = os.path.join(BASE_PATH, "annotations/annotations averaged per song/song_level/static_annotations_averaged_songs_1_2000.csv")
ANNOTATIONS_PATH_2 = os.path.join(BASE_PATH, "annotations/annotations averaged per song/song_level/static_annotations_averaged_songs_2000_2058.csv")
FEATURES_DIR = os.path.join(BASE_PATH, "features")
MODELS_DIR = "src/models"
RESULTS_DIR = "results"
EVALUATION_FILE = os.path.join(RESULTS_DIR, "model_evaluations.json")

# Ensure directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_dataset(sample_size=1200):
    """Load and merge DEAM features and annotations."""
    # Load annotations
    df1 = pd.read_csv(ANNOTATIONS_PATH_1, skipinitialspace=True)
    df2 = pd.read_csv(ANNOTATIONS_PATH_2, skipinitialspace=True)
    annotations_df = pd.concat([df1, df2], ignore_index=True)
    
    # Normalize to 0-1 scale
    annotations_df['valence'] = (annotations_df['valence_mean'] - 1) / 8
    annotations_df['arousal'] = (annotations_df['arousal_mean'] - 1) / 8
    
    # Load features
    available_files = [f for f in os.listdir(FEATURES_DIR) if f.endswith('.csv')]
    song_ids = sorted([int(f.replace('.csv', '')) for f in available_files])[:sample_size]
    
    feature_data = []
    for song_id in song_ids:
        file_path = os.path.join(FEATURES_DIR, f"{song_id}.csv")
        try:
            df = pd.read_csv(file_path, delimiter=';')
            # Aggregate features (mean and std)
            feature_vector = [df[col].mean() for col in df.columns[1:]] + [df[col].std() for col in df.columns[1:]]
            feature_row = {'song_id': song_id}
            for i, val in enumerate(feature_vector):
                feature_row[f'feature_{i}'] = val
            feature_data.append(feature_row)
        except Exception as e:
            print(f"Skipping {song_id}.csv due to error: {e}")

    features_df = pd.DataFrame(feature_data).fillna(0)
    
    # Merge datasets
    merged = features_df.merge(annotations_df[['song_id', 'valence', 'arousal']], on='song_id')
    feature_cols = [col for col in merged.columns if col.startswith('feature_')]
    X = merged[feature_cols]
    y = merged[['valence', 'arousal']]
    
    return X, y

def evaluate_model(y_true, y_pred):
    """Calculate and return regression metrics."""
    metrics = {
        'r2_score': r2_score(y_true, y_pred),
        'mse': mean_squared_error(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred)
    }
    return metrics

def main():
    """Main training and evaluation pipeline."""
    print("1. Loading dataset...")
    X, y = load_dataset()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler
    with open(os.path.join(MODELS_DIR, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)

    print(f"Dataset loaded: {len(X_train)} training samples, {len(X_test)} testing samples.")

    models_to_train = {
        "Ridge": Ridge(alpha=10.0),
        "SVR": SVR(C=10, kernel='rbf'),
        "XGBoost": xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42),
        "MLP": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, alpha=0.1, random_state=42, early_stopping=True)
    }
    
    all_evaluations = {}

    for name, model in models_to_train.items():
        print(f"\n2. Training {name} model...")
        
        # Train separate models for valence and arousal
        valence_model = model
        valence_model.fit(X_train_scaled, y_train['valence'])
        
        arousal_model = model
        arousal_model.fit(X_train_scaled, y_train['arousal'])
        
        # Save models
        with open(os.path.join(MODELS_DIR, f'{name}_valence.pkl'), 'wb') as f:
            pickle.dump(valence_model, f)
        with open(os.path.join(MODELS_DIR, f'{name}_arousal.pkl'), 'wb') as f:
            pickle.dump(arousal_model, f)
            
        print(f"   {name} models saved to {MODELS_DIR}/")

        # Evaluate models
        print(f"3. Evaluating {name} model...")
        y_pred_valence = valence_model.predict(X_test_scaled)
        y_pred_arousal = arousal_model.predict(X_test_scaled)
        
        valence_metrics = evaluate_model(y_test['valence'], y_pred_valence)
        arousal_metrics = evaluate_model(y_test['arousal'], y_pred_arousal)
        
        all_evaluations[name] = {
            "valence": valence_metrics,
            "arousal": arousal_metrics,
            "training_date": datetime.now().isoformat()
        }
        print(f"   Valence Metrics: {valence_metrics}")
        print(f"   Arousal Metrics: {arousal_metrics}")

    # Save all evaluations to a single file
    with open(EVALUATION_FILE, 'w') as f:
        json.dump(all_evaluations, f, indent=4)
        
    print(f"\n4. All model evaluations saved to {EVALUATION_FILE}")
    print("\nTraining and evaluation complete!")

if __name__ == "__main__":
    main()
