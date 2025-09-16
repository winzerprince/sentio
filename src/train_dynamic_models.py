"""
Train, Evaluate, and Save Emotion Analysis Models using DYNAMIC annotations.

This script trains multiple regression models on the DEAM dataset using
per-second (dynamic) annotations. It evaluates their performance and saves
the trained models and evaluation metrics for later analysis.
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
KAGGLE_ENV = os.path.exists('/kaggle/input')

if KAGGLE_ENV:
    print("Kaggle environment detected. Adjusting paths.")
    BASE_PATH = "/kaggle/input/deam-music-emotion-dataset/DEAM"
    # Output files must be saved to /kaggle/working/
    MODELS_DIR = "/kaggle/working/models/dynamic"
    RESULTS_DIR = "/kaggle/working/results"
else:
    print("Local environment detected.")
    BASE_PATH = "dataset/DEAM"
    MODELS_DIR = "src/models/dynamic" # New directory for dynamic models
    RESULTS_DIR = "results"

DYNAMIC_ANNOTATIONS_DIR = os.path.join(BASE_PATH, "annotations/annotations averaged per song/dynamic (per second annotations)")
FEATURES_DIR = os.path.join(BASE_PATH, "features")
EVALUATION_FILE = os.path.join(RESULTS_DIR, "dynamic_model_evaluations.json")

# Ensure directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_dynamic_dataset(sample_size=100):
    """
    Load and merge DEAM features and DYNAMIC (per-second) annotations.
    Each time frame becomes a sample.
    """
    print("Loading dynamic annotations...")
    arousal_df = pd.read_csv(os.path.join(DYNAMIC_ANNOTATIONS_DIR, 'arousal.csv'))
    valence_df = pd.read_csv(os.path.join(DYNAMIC_ANNOTATIONS_DIR, 'valence.csv'))

    # Melt the dataframes to a long format
    arousal_long = arousal_df.melt(id_vars=['Sample'], var_name='song_id', value_name='arousal').rename(columns={'Sample': 'time'})
    valence_long = valence_df.melt(id_vars=['Sample'], var_name='song_id', value_name='valence').rename(columns={'Sample': 'time'})

    # Convert song_id to integer
    arousal_long['song_id'] = arousal_long['song_id'].astype(int)
    valence_long['song_id'] = valence_long['song_id'].astype(int)

    # Merge arousal and valence
    annotations_df = pd.merge(arousal_long, valence_long, on=['song_id', 'time']).dropna()
    
    # Normalize to 0-1 scale
    annotations_df['valence'] = (annotations_df['valence'] - 1) / 8
    annotations_df['arousal'] = (annotations_df['arousal'] - 1) / 8
    
    print(f"Loaded {len(annotations_df)} total annotation frames.")

    # Load features and align them with annotations
    song_ids_to_process = sorted(annotations_df['song_id'].unique())[:sample_size]
    print(f"Processing features for {len(song_ids_to_process)} songs...")
    
    all_frames = []
    for song_id in song_ids_to_process:
        feature_path = os.path.join(FEATURES_DIR, f"{song_id}.csv")
        if not os.path.exists(feature_path):
            continue
            
        features_df = pd.read_csv(feature_path, delimiter=';')
        # The 'time' in features is in seconds, matching the annotations 'time'
        features_df['time'] = features_df['time'].round().astype(int)
        
        song_annotations = annotations_df[annotations_df['song_id'] == song_id]
        
        # Merge features with annotations for this song
        merged_song_df = pd.merge(features_df, song_annotations, on='time', how='inner')
        all_frames.append(merged_song_df)

    if not all_frames:
        raise ValueError("No dataframes to concatenate. Check sample_size and file paths.")

    final_df = pd.concat(all_frames, ignore_index=True)
    
    feature_cols = [col for col in final_df.columns if col not in ['song_id', 'time', 'valence', 'arousal']]
    X = final_df[feature_cols]
    y = final_df[['valence', 'arousal']]
    
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
    """Main training and evaluation pipeline for dynamic models."""
    print("--- Training Models on Dynamic Per-Second Annotations ---")
    # Using a smaller sample size initially due to the large number of frames
    print("1. Loading dataset...")
    try:
        X, y = load_dynamic_dataset(sample_size=200) # Limit to 200 songs to manage memory
    except ValueError as e:
        print(f"Error: {e}")
        return

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler
    with open(os.path.join(MODELS_DIR, 'dynamic_scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)

    print(f"Dataset loaded: {len(X_train)} training frames, {len(X_test)} testing frames.")

    # NOTE: Using models with parameters that are faster to train given the larger dataset
    models_to_train = {
        "Ridge": Ridge(alpha=10.0),
        "SVR": SVR(C=1, kernel='rbf'), # Reduced C for faster training
        "XGBoost": xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42), # Shallower trees
        "MLP": MLPRegressor(hidden_layer_sizes=(64,), max_iter=300, alpha=0.1, random_state=42, early_stopping=True) # Simpler network
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
        with open(os.path.join(MODELS_DIR, f'{name}_valence_dynamic.pkl'), 'wb') as f:
            pickle.dump(valence_model, f)
        with open(os.path.join(MODELS_DIR, f'{name}_arousal_dynamic.pkl'), 'wb') as f:
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
            "training_date": datetime.now().isoformat(),
            "songs_sampled": 200,
            "training_frames": len(X_train)
        }
        print(f"   Valence Metrics: {valence_metrics}")
        print(f"   Arousal Metrics: {arousal_metrics}")

    # Save all evaluations to a single file
    with open(EVALUATION_FILE, 'w') as f:
        json.dump(all_evaluations, f, indent=4)
        
    print(f"\n4. All dynamic model evaluations saved to {EVALUATION_FILE}")
    print("\nDynamic training and evaluation complete!")

if __name__ == "__main__":
    main()
