"""
Evaluation Utilities for Emotion Prediction Models

This module provides tools for evaluating emotion prediction models:
1. Model evaluation functions and metrics
2. Visualization of prediction results
3. Model comparison and selection
4. Cross-validation strategies

Key components:
- evaluate_models: Evaluate multiple models using consistent metrics
- plot_results: Visualize model performance comparisons
- visualize_predictions: Plot actual vs. predicted emotions
- cross_validate: Perform cross-validation for model evaluation

Notes:
- These utilities help compare the performance of different model types
- Important metrics include R², RMSE, and MAE for regression evaluation
- Visualization aids in understanding model strengths and weaknesses
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score, KFold
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_model(model, X, y, emotion_dims):
    """
    Evaluate a trained model on test data.
    
    Args:
        model: Trained model with predict method
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
        
        return pd.DataFrame(results)
        
    except Exception as e:
        logger.error(f"Error during model evaluation: {e}")
        raise
        
def cross_validate_model(model, X, y, emotion_dims, cv=5):
    """
    Perform cross-validation for model evaluation.
    
    Args:
        model: Model with fit and predict methods
        X: Feature matrix
        y: Target values
        emotion_dims: Names of emotion dimensions
        cv: Number of cross-validation folds
        
    Returns:
        DataFrame with cross-validation results
    """
    try:
        cv_results = []
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        
        # For each emotion dimension
        for i, dim in enumerate(emotion_dims):
            fold_metrics = []
            
            # For each fold
            for train_idx, test_idx in kf.split(X):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx, i:i+1], y[test_idx, i:i+1]
                
                # Train model
                model.fit(X_train, y_train)
                
                # Evaluate
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                fold_metrics.append({
                    'MSE': mse,
                    'RMSE': np.sqrt(mse),
                    'R²': r2
                })
            
            # Calculate mean and std of metrics across folds
            metrics_df = pd.DataFrame(fold_metrics)
            mean_metrics = metrics_df.mean()
            std_metrics = metrics_df.std()
            
            cv_results.append({
                'Dimension': dim,
                'MSE_mean': mean_metrics['MSE'],
                'MSE_std': std_metrics['MSE'],
                'RMSE_mean': mean_metrics['RMSE'],
                'RMSE_std': std_metrics['RMSE'],
                'R²_mean': mean_metrics['R²'],
                'R²_std': std_metrics['R²']
            })
        
        return pd.DataFrame(cv_results)
        
    except Exception as e:
        logger.error(f"Error during cross-validation: {e}")
        raise
        
def plot_model_comparison(results_df, metric='R²', output_dir="results"):
    """
    Plot comparison of model performance.
    
    Args:
        results_df: DataFrame with model comparison results
        metric: Metric to plot (e.g., 'R²', 'RMSE', 'MSE')
        output_dir: Directory to save plots
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        plt.figure(figsize=(12, 8))
        
        # Create grouped bar plot
        sns.barplot(x='Dimension', y=metric, hue='Model', data=results_df)
        
        plt.title(f"Model Comparison - {metric}")
        plt.xlabel("Emotion Dimension")
        plt.ylabel(metric)
        
        if metric == 'R²':
            plt.ylim(0, 1)  # R² is typically between 0 and 1
            
        plt.legend(title="Model Type")
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save the plot
        safe_metric = metric.lower().replace('²', '2')
        plt.savefig(f"{output_dir}/model_comparison_{safe_metric}.png")
        plt.close()
        
        logger.info(f"Model comparison plot for {metric} saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Error plotting model comparison: {e}")
        raise
        
def visualize_predictions(model_dict, X, y, emotion_dims, n_samples=10, output_dir="results"):
    """
    Visualize predictions from multiple models for comparison.
    
    Args:
        model_dict: Dictionary mapping model names to model objects
        X: Feature matrix
        y: Target values
        emotion_dims: Names of emotion dimensions
        n_samples: Number of samples to visualize
        output_dir: Directory to save plots
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Select random samples for visualization
        if n_samples < X.shape[0]:
            sample_indices = np.random.choice(X.shape[0], n_samples, replace=False)
            X_sample = X[sample_indices]
            y_sample = y[sample_indices]
        else:
            X_sample = X
            y_sample = y
            sample_indices = np.arange(X.shape[0])
        
        # For each emotion dimension
        for dim_idx, dim_name in enumerate(emotion_dims):
            plt.figure(figsize=(14, 8))
            
            # Plot actual values
            plt.scatter(range(len(y_sample)), y_sample[:, dim_idx], 
                      marker='o', s=100, label='Actual', color='black')
            
            # Plot predictions for each model with different colors
            for model_name, model in model_dict.items():
                y_pred = model.predict(X_sample)
                plt.plot(range(len(y_pred)), y_pred[:, dim_idx], 
                       'o-', label=f"{model_name} Predicted", alpha=0.7)
            
            plt.title(f"Actual vs. Predicted - {dim_name}")
            plt.xlabel("Sample Index")
            plt.ylabel(f"{dim_name} Value")
            plt.legend()
            plt.grid(True)
            
            # Add sample indices to x-axis for reference
            plt.xticks(range(len(y_sample)), sample_indices)
            
            # Save the plot
            safe_dim = dim_name.lower().replace(' ', '_')
            plt.savefig(f"{output_dir}/predictions_comparison_{safe_dim}.png")
            plt.close()
        
        logger.info(f"Prediction visualization plots saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Error visualizing predictions: {e}")
        raise

def plot_learning_curves(model_sizes, train_scores, test_scores, title, output_dir="results"):
    """
    Plot learning curves showing model performance vs. training data size.
    
    Args:
        model_sizes: List of training set sizes
        train_scores: List of training scores for each size
        test_scores: List of test scores for each size
        title: Plot title
        output_dir: Directory to save plots
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        plt.figure(figsize=(10, 6))
        plt.plot(model_sizes, train_scores, 'o-', color='r', label='Training score')
        plt.plot(model_sizes, test_scores, 'o-', color='g', label='Test score')
        
        plt.title(title)
        plt.xlabel('Training Set Size')
        plt.ylabel('Score')
        plt.legend(loc='best')
        plt.grid(True)
        
        # Save the plot
        safe_title = title.lower().replace(' ', '_')
        plt.savefig(f"{output_dir}/learning_curve_{safe_title}.png")
        plt.close()
        
        logger.info(f"Learning curve plot saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Error plotting learning curves: {e}")
        raise
