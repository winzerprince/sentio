"""
Main Module for Emotion Prediction Models

This script serves as the entry point for training and evaluating emotion prediction models.
The module includes:

1. Command-line argument parsing
2. Data loading and preprocessing
3. Model training for different algorithms
4. Evaluation and result visualization

Usage:
    python main.py --annotations path/to/annotations.csv --features path/to/features 
                  --models linear svr xgboost --dynamic

Key components:
- train_models: Train multiple model types on emotion data
- evaluate_models: Evaluate and compare model performance
- save_results: Save trained models and evaluation results
- main: Entry point with command-line argument parsing

Notes:
- Models are trained on either static or dynamic emotion data
- Results are saved for future use and comparison
- Command-line interface enables flexible usage
"""

import os
import argparse
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Import custom modules
from data_processing.data_loader import DataLoader
from models.linear_model import LinearModel
from models.svr_model import SVRModel
from models.xgboost_model import XGBoostModel
from utils.evaluate import evaluate_model, plot_model_comparison, visualize_predictions
from utils.plot_metrics import plot_feature_importance

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"logs/training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

def train_models(annotations_path, features_dir, model_types, use_dynamic=False, output_dir="models"):
    """
    Train models on emotion data.
    
    Args:
        annotations_path: Path to annotations file
        features_dir: Directory containing feature files
        model_types: List of model types to train
        use_dynamic: Whether to use dynamic emotion data
        output_dir: Directory to save model outputs
    
    Returns:
        Tuple of (trained models dict, feature matrix, target values, emotion dimensions)
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        
        # Load and prepare data
        logger.info(f"Loading data from annotations: {annotations_path}, features: {features_dir}")
        data_loader = DataLoader(annotations_path, features_dir)
        X, y, song_ids = data_loader.merge_data(use_dynamic=use_dynamic)
        
        # Split data before scaling
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        logger.info("Features scaled using StandardScaler")
        
        # Save the scaler
        scaler_path = os.path.join(output_dir, "scaler.pkl")
        joblib.dump(scaler, scaler_path)
        logger.info(f"Scaler saved to {scaler_path}")

        # Define emotion dimensions based on the actual data loaded
        # Extract emotion dimension names from the actual data columns used
        if use_dynamic:
            # For dynamic data, get the actual column names used
            emotion_dims = ["dynamic_valence", "dynamic_arousal"]  # Based on typical dynamic annotations
        else:
            # For static data, get the actual column names used (valence_mean, arousal_mean)
            emotion_dims = ["valence", "arousal"]  # Based on the loaded static annotations
        
        logger.info(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"Target shape: {y.shape}")
        logger.info(f"Emotion dimensions: {emotion_dims}")
        
        # Dictionary to store trained models
        trained_models = {}
        
        # Train each model type
        for model_type in model_types:
            logger.info(f"Training {model_type} model for {'dynamic' if use_dynamic else 'static'} emotion prediction")
            
            # Create model based on type
            if model_type == "linear":
                model = LinearModel(use_ridge=True, is_dynamic=use_dynamic)
            elif model_type == "svr":
                model = SVRModel(kernel='rbf', is_dynamic=use_dynamic)
            elif model_type == "xgboost":
                model = XGBoostModel(is_dynamic=use_dynamic)
            else:
                logger.error(f"Unknown model type: {model_type}")
                continue
                
            # Train the model
            try:
                model.train(X_train_scaled, y_train, emotion_dims=emotion_dims)
                trained_models[model_type] = model
                
                # Save the trained model
                model_save_dir = os.path.join(output_dir, model_type)
                os.makedirs(model_save_dir, exist_ok=True)
                model.save(model_dir=model_save_dir)
                logger.info(f"{model_type} model trained and saved to {model_save_dir}")
                
            except Exception as e:
                logger.error(f"Error training {model_type} model: {e}")
        
        return trained_models, X_test_scaled, y_test, emotion_dims
        
    except Exception as e:
        logger.error(f"Error in train_models: {e}")
        raise

def evaluate_models(models_dict, X_test, y_test, emotion_dims, output_dir="results"):
    """
    Evaluate and compare trained models.
    
    Args:
        models_dict: Dictionary of trained models
        X_test: Test feature matrix (already scaled)
        y_test: Test target values
        emotion_dims: Names of emotion dimensions
        output_dir: Directory to save evaluation results
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Results for each model
        all_results = []
        
        for model_name, model in models_dict.items():
            logger.info(f"Evaluating {model_name} model")
            
            # Evaluate model
            results = evaluate_model(model, X_test, y_test, emotion_dims)
            results['Model'] = model_name
            all_results.append(results)
            
            # Save individual results
            results.to_csv(f"{output_dir}/{model_name}_metrics.csv", index=False)
        
        # Combine all results for comparison
        if all_results:
            combined_results = pd.concat(all_results)
            combined_results.to_csv(f"{output_dir}/all_models_comparison.csv", index=False)
            
            # Plot comparison
            for metric in ['R²', 'RMSE', 'MAE']:
                plot_model_comparison(combined_results, metric=metric, output_dir=output_dir)
            
            # Visualize predictions
            visualize_predictions(models_dict, X_test, y_test, emotion_dims, n_samples=10, output_dir=output_dir)
            
            logger.info(f"Evaluation results saved to {output_dir}")
            
            return combined_results
            
    except Exception as e:
        logger.error(f"Error evaluating models: {e}")
        raise

def main():
    """Main function for training and evaluating emotion prediction models."""
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train and evaluate emotion regression models")
    parser.add_argument("--annotations", type=str, required=True, 
                        help="Path to annotations file")
    parser.add_argument("--features", type=str, default="selected", 
                        help="Directory containing feature files")
    parser.add_argument("--models", type=str, nargs="+", default=["linear", "svr", "xgboost"], 
                        choices=["linear", "svr", "xgboost"], help="Model types to train")
    parser.add_argument("--dynamic", action="store_true", 
                        help="Use dynamic emotion data")
    parser.add_argument("--output", type=str, default="output", 
                        help="Directory for output files")
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.output, exist_ok=True)
    models_dir = os.path.join(args.output, "models")
    results_dir = os.path.join(args.output, "results")
    
    logger.info("Starting emotion prediction model training and evaluation")
    logger.info(f"Arguments: {args}")
    
    # Train models
    trained_models, X_test_scaled, y_test, emotion_dims = train_models(
        args.annotations, args.features, args.models, 
        use_dynamic=args.dynamic, output_dir=models_dir
    )
    
    # Evaluate models
    evaluate_models(trained_models, X_test_scaled, y_test, emotion_dims, output_dir=results_dir)
    
    logger.info("Model training and evaluation completed")

if __name__ == "__main__":
    main()
