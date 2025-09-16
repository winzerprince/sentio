"""
Plot Utility Functions for Emotion Prediction

This module provides visualization tools for emotion prediction models:
1. Various plots for model evaluation
2. Emotion visualization over time 
3. Feature importance visualization
4. Confusion matrices and heatmaps

Key components:
- plot_confusion_matrix: Create confusion matrices for emotion classification
- plot_feature_importance: Visualize important features for prediction
- plot_emotion_over_time: Visualize dynamic emotions across time
- plot_training_history: Show model training metrics over epochs

Notes:
- Visualizations help interpret model performance and predictions
- Time-based visualizations are particularly useful for dynamic emotion modeling
- Feature importance plots help identify which audio features predict emotions
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def plot_feature_importance(feature_names, importances, top_n=20, title="Feature Importance", output_dir="results"):
    """
    Plot feature importance for a model.
    
    Args:
        feature_names: List of feature names
        importances: Array of importance values
        top_n: Number of top features to show
        title: Plot title
        output_dir: Directory to save plots
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Sort features by importance
        indices = np.argsort(importances)[::-1]
        
        # Select top N features
        top_indices = indices[:top_n]
        top_features = [feature_names[i] if i < len(feature_names) else f"feature_{i}" for i in top_indices]
        top_importances = importances[top_indices]
        
        # Create plot
        plt.figure(figsize=(12, 8))
        plt.title(title)
        plt.barh(range(len(top_features)), top_importances, align='center')
        plt.yticks(range(len(top_features)), top_features)
        plt.xlabel('Importance')
        plt.tight_layout()
        
        # Save the plot
        safe_title = title.lower().replace(' ', '_')
        plt.savefig(f"{output_dir}/{safe_title}.png")
        plt.close()
        
        # Also save as CSV
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        importance_df.to_csv(f"{output_dir}/{safe_title}.csv", index=False)
        
        logger.info(f"Feature importance plot saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Error plotting feature importance: {e}")
        raise

def plot_emotion_over_time(times, emotions, emotion_names, title="Emotions Over Time", output_dir="results"):
    """
    Plot emotion values over time (for dynamic emotion prediction).
    
    Args:
        times: Array of time points
        emotions: Array of emotion values (rows=time, cols=emotions)
        emotion_names: List of emotion dimension names
        title: Plot title
        output_dir: Directory to save plots
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        plt.figure(figsize=(14, 8))
        
        # Plot each emotion dimension
        for i, emotion_name in enumerate(emotion_names):
            plt.plot(times, emotions[:, i], '-', label=emotion_name)
        
        plt.title(title)
        plt.xlabel('Time (seconds)')
        plt.ylabel('Emotion Value')
        plt.legend()
        plt.grid(True)
        
        # Save the plot
        safe_title = title.lower().replace(' ', '_')
        plt.savefig(f"{output_dir}/{safe_title}.png")
        plt.close()
        
        logger.info(f"Emotion over time plot saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Error plotting emotions over time: {e}")
        raise

def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title='Confusion Matrix', 
                        output_dir="results"):
    """
    Plot confusion matrix for classification tasks.
    
    Args:
        y_true: True class labels
        y_pred: Predicted class labels
        classes: List of class names
        normalize: Whether to normalize the confusion matrix
        title: Plot title
        output_dir: Directory to save plots
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                  xticklabels=classes, yticklabels=classes, cmap='Blues')
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        # Save the plot
        safe_title = title.lower().replace(' ', '_')
        plt.savefig(f"{output_dir}/{safe_title}.png")
        plt.close()
        
        logger.info(f"Confusion matrix plot saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Error plotting confusion matrix: {e}")
        raise

def plot_training_history(history, metrics=['loss'], title='Training History', output_dir="results"):
    """
    Plot training metrics over epochs.
    
    Args:
        history: Dictionary containing training history (keys are metric names, values are lists of values per epoch)
        metrics: List of metrics to plot
        title: Plot title
        output_dir: Directory to save plots
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        plt.figure(figsize=(12, 6))
        
        # Plot each metric
        for metric in metrics:
            if metric in history:
                plt.plot(history[metric], '-', label=f'Training {metric}')
                
            # If validation metric exists
            val_metric = f'val_{metric}'
            if val_metric in history:
                plt.plot(history[val_metric], '--', label=f'Validation {metric}')
        
        plt.title(title)
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        
        # Save the plot
        safe_title = title.lower().replace(' ', '_')
        plt.savefig(f"{output_dir}/{safe_title}.png")
        plt.close()
        
        logger.info(f"Training history plot saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Error plotting training history: {e}")
        raise

def plot_distribution(data, labels=None, title='Distribution', output_dir="results"):
    """
    Plot distribution of data (e.g., emotion values).
    
    Args:
        data: Array of data to plot
        labels: Optional labels for multiple distributions
        title: Plot title
        output_dir: Directory to save plots
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        plt.figure(figsize=(12, 6))
        
        if labels is not None and len(data) == len(labels):
            # Plot multiple distributions
            for i, (d, label) in enumerate(zip(data, labels)):
                sns.kdeplot(d, label=label)
        else:
            # Plot single distribution
            sns.histplot(data, kde=True)
        
        plt.title(title)
        plt.xlabel('Value')
        plt.ylabel('Density')
        if labels is not None:
            plt.legend()
        plt.grid(True)
        
        # Save the plot
        safe_title = title.lower().replace(' ', '_')
        plt.savefig(f"{output_dir}/{safe_title}.png")
        plt.close()
        
        logger.info(f"Distribution plot saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Error plotting distribution: {e}")
        raise
