"""
Sample Prediction Script

This script demonstrates how to use the trained models to make predictions on new, unseen data samples.

IMPORTANT DISCOVERY:
During training, the annotation scaling from [1,9] to [-1,1] was NOT applied because:
- The DataLoader._scale_annotations() function looks for columns named 'valence' and 'arousal' 
- But the actual annotation file has columns named 'valence_mean' and 'arousal_mean'
- Therefore, the models were trained on the original [1,9] scale, not the intended [-1,1] scale

This script handles this by:
1. Predicting on the original [1,9] scale (as the models were actually trained)
2. Converting predictions to [-1,1] scale for interpretation
3. Providing emotion interpretations based on the scaled values
"""
import os
import joblib
import pandas as pd
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_static_features(feature_df):
    """
    Process a single feature file into a static feature vector.
    This exactly mimics the logic in DataLoader._process_static_features for static features.
    """
    # For static emotions, we need to aggregate features
    # This matches the exact aggregation used in training
    agg_features = []
    
    # Add mean of each numerical column
    means = feature_df.mean(numeric_only=True)
    agg_features.extend(means.values)
    
    # Add standard deviation of each numerical column
    stds = feature_df.std(numeric_only=True)
    agg_features.extend(stds.values)
    
    # Add min/max of each numerical column for range
    mins = feature_df.min(numeric_only=True)
    maxs = feature_df.max(numeric_only=True)
    agg_features.extend(mins.values)
    agg_features.extend(maxs.values)
    
    # Return aggregated features as a flat array
    return np.array(agg_features).reshape(1, -1)

def predict_emotion(sample_feature_path, models_dir='output/models'):
    """
    Predicts valence and arousal for a given feature file.
    """
    try:
        # 1. Load the scaler
        scaler_path = os.path.join(models_dir, 'scaler.pkl')
        scaler = joblib.load(scaler_path)
        logger.info(f"Scaler loaded from {scaler_path}")

        # 2. Load the models
        valence_model_path = os.path.join(models_dir, 'xgboost', 'xgboost_static_valence.joblib')
        arousal_model_path = os.path.join(models_dir, 'xgboost', 'xgboost_static_arousal.joblib')
        
        valence_model = joblib.load(valence_model_path)
        arousal_model = joblib.load(arousal_model_path)
        logger.info("XGBoost models for valence and arousal loaded.")

        # 3. Load and process the sample data
        if not os.path.exists(sample_feature_path):
            logger.error(f"Sample feature file not found: {sample_feature_path}")
            return

        feature_df = pd.read_csv(sample_feature_path)
        feature_vector = process_static_features(feature_df)
        logger.info(f"Processed sample data from {sample_feature_path}. Feature vector shape: {feature_vector.shape}")

        # 4. Scale the features
        scaled_features = scaler.transform(feature_vector)
        logger.info("Sample features scaled.")

        # 5. Make predictions
        predicted_valence = valence_model.predict(scaled_features)
        predicted_arousal = arousal_model.predict(scaled_features)

        # The output is on a [-1, 1] scale.
        logger.info("Predictions made.")
        
        print("\n--- Emotion Prediction Results ---")
        print(f"Sample File: {sample_feature_path}")
        print(f"Predicted Valence: {predicted_valence[0]:.4f} (on original 1-9 scale)")
        print(f"Predicted Arousal: {predicted_arousal[0]:.4f} (on original 1-9 scale)")
        
        # Convert to scaled [-1, 1] range for interpretation
        scaled_valence = (predicted_valence[0] - 5.0) / 4.0
        scaled_arousal = (predicted_arousal[0] - 5.0) / 4.0
        
        print(f"Scaled Valence: {scaled_valence:.4f} (on -1 to 1 scale)")
        print(f"Scaled Arousal: {scaled_arousal:.4f} (on -1 to 1 scale)")
        
        # Interpretation
        valence_interpretation = "positive" if scaled_valence > 0.1 else "negative" if scaled_valence < -0.1 else "neutral"
        arousal_interpretation = "high energy" if scaled_arousal > 0.1 else "low energy" if scaled_arousal < -0.1 else "moderate energy"
        
        print(f"Interpretation: {valence_interpretation} emotion, {arousal_interpretation}")
        print("----------------------------------\n")

    except FileNotFoundError as e:
        logger.error(f"Error loading model or scaler: {e}. Make sure you have run the training script first.")
    except Exception as e:
        logger.error(f"An error occurred during prediction: {e}")

if __name__ == '__main__':
    # Test with multiple sample files from the 'selected' directory
    import glob
    
    sample_files = glob.glob('selected/*_selected.csv')[:7]  # Get first 5 files
    
    print("Testing emotion predictions on multiple samples:\n")
    
    for sample_file in sample_files:
        predict_emotion(sample_file)
        
    print("\nNote: Values outside [-1, 1] range may indicate the model needs further calibration.")
