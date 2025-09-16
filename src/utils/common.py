"""
Common Utilities for Emotion Prediction Models

This module provides shared functionality used across different model types.
The module includes:

1. Model evaluation functions
2. Data preprocessing utilities
3. Visualization tools for model comparison
4. Feature selection tools

Key components:
- evaluate_model: Standard evaluation of emotion prediction models
- compare_models: Compare performance across different model types
- plot_predictions: Visualize actual vs. predicted emotions
- select_features: Feature selection to improve model performance

Notes:
- These utilities ensure consistent evaluation across model types
- Standard metrics include RMSE, MSE, and R² for regression tasks
- Visualization helps interpret model performance and emotion predictions
"""

import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression
import joblib
import logging
import torch  # Keep the existing torch import

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Keep existing PyTorch functions
def save_model(model, path):
    """Saves a PyTorch model."""
    torch.save(model.state_dict(), path)

def load_model(model, path, device):
    """Loads a PyTorch model."""
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Add new functions for regression models
def evaluate_model(model, X, y, emotion_dims):
    """
    Evaluate a trained model on test data.
    
    Args:
        model: Trained regression model with predict method
        X: Feature matrix
        y: Target values
        emotion_dims: Names of emotion dimensions
        
    Returns:
        DataFrame with evaluation metrics
    """
    try:
        # Make predictions
        y_pred = model.predict(X)
        
        # Calculate metrics for each emotion dimension
        results = []
        
        for i, dim in enumerate(emotion_dims):
            mse = mean_squared_error(y[:, i], y_pred[:, i])
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y[:, i], y_pred[:, i])
            r2 = r2_score(y[:, i], y_pred[:, i])
            
            results.append({
                'Dimension': dim,
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'R²': r2
            })
        
        # Convert to DataFrame for easy handling
        return pd.DataFrame(results)
        
    except Exception as e:
        logger.error(f"Error during model evaluation: {e}")
        raise

def compare_models(models_dict, X, y, emotion_dims, output_dir="results"):
    """
    Compare multiple models on the same dataset.
    
    Args:
        models_dict: Dictionary mapping model names to model objects
        X: Feature matrix
        y: Target values
        emotion_dims: Names of emotion dimensions
        output_dir: Directory to save results
        
    Returns:
        DataFrame with comparative results
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        all_results = []
        
        # Evaluate each model
        for model_name, model in models_dict.items():
            logger.info(f"Evaluating model: {model_name}")
            results = evaluate_model(model, X, y, emotion_dims)
            results['Model'] = model_name
            all_results.append(results)
        
        # Combine all results
        if all_results:
            combined_results = pd.concat(all_results)
            
            # Save results to CSV
            combined_results.to_csv(f"{output_dir}/model_comparison.csv", index=False)
            
            # Create plots for each metric
            for metric in ['R²', 'RMSE', 'MAE']:
                plt.figure(figsize=(12, 8))
                sns.barplot(x='Dimension', y=metric, hue='Model', data=combined_results)
                plt.title(f"Model Comparison - {metric}")
                
                if metric == 'R²':
                    plt.ylim(0, 1)  # R² should typically be between 0 and 1
                    
                plt.tight_layout()
                plt.savefig(f"{output_dir}/comparison_{metric.lower().replace('²', '2')}.png")
                plt.close()
            
            logger.info(f"Comparison results saved to {output_dir}")
            return combined_results
        
        return None
        
    except Exception as e:
        logger.error(f"Error during model comparison: {e}")
        raise

def plot_predictions(model, X, y, emotion_dims, n_samples=20, output_dir="results"):
    """
    Plot actual vs. predicted values for visual comparison.
    
    Args:
        model: Trained regression model with predict method
        X: Feature matrix
        y: Target values
        emotion_dims: Names of emotion dimensions
        n_samples: Number of samples to display
        output_dir: Directory to save results
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Make predictions
        y_pred = model.predict(X)
        
        # Limit to a subset of samples for clarity
        if n_samples < X.shape[0]:
            indices = np.random.choice(X.shape[0], n_samples, replace=False)
            y_subset = y[indices]
            y_pred_subset = y_pred[indices]
        else:
            y_subset = y
            y_pred_subset = y_pred
            
        # Plot each emotion dimension
        for i, dim in enumerate(emotion_dims):
            plt.figure(figsize=(12, 6))
            
            # Plot actual vs predicted
            plt.scatter(range(len(y_subset)), y_subset[:, i], label='Actual', marker='o')
            plt.scatter(range(len(y_pred_subset)), y_pred_subset[:, i], label='Predicted', marker='x')
            
            plt.title(f"Actual vs Predicted - {dim}")
            plt.xlabel("Sample Index")
            plt.ylabel(f"{dim} Value")
            plt.legend()
            plt.grid(True)
            
            # Save the plot
            safe_dim = dim.lower().replace(' ', '_')
            plt.savefig(f"{output_dir}/predictions_{safe_dim}.png")
            plt.close()
            
        logger.info(f"Prediction plots saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Error plotting predictions: {e}")
        raise
        
def select_features(X, y, k=20):
    """
    Select the k best features for each emotion dimension.
    
    Args:
        X: Feature matrix
        y: Target values (one column per emotion dimension)
        k: Number of features to select
        
    Returns:
        Dictionary mapping emotion dimensions to selected feature indices
    """
    try:
        selected_features = {}
        
        # For each emotion dimension
        for i in range(y.shape[1]):
            # Select k best features
            selector = SelectKBest(f_regression, k=k)
            selector.fit(X, y[:, i])
            
            # Get the indices of selected features
            selected = selector.get_support(indices=True)
            selected_features[i] = selected
            
            logger.info(f"Selected {len(selected)} features for dimension {i}")
            
        return selected_features
        
    except Exception as e:
        logger.error(f"Error during feature selection: {e}")
        raise

def save_regression_model_results(model, metrics, predictions, features, output_dir="results", model_name="model"):
    """
    Save regression model results, metrics and predictions.
    
    Args:
        model: The trained model
        metrics: DataFrame of evaluation metrics
        predictions: Model predictions
        features: Feature information (optional)
        output_dir: Directory to save results
        model_name: Name to use for saved files
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model
        joblib.dump(model, f"{output_dir}/{model_name}.joblib")
        
        # Save metrics
        if metrics is not None:
            metrics.to_csv(f"{output_dir}/{model_name}_metrics.csv", index=False)
        
        # Save predictions
        if predictions is not None:
            np.save(f"{output_dir}/{model_name}_predictions.npy", predictions)
        
        # Save feature information
        if features is not None:
            joblib.dump(features, f"{output_dir}/{model_name}_features.joblib")
            
        logger.info(f"Model results saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Error saving model results: {e}")
        raise
