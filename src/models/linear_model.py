"""
Linear Regression Model for Emotion Prediction

This module implements linear regression models for predicting emotions from audio features.
The module includes:

1. Basic linear regression models
2. Ridge regression for regularization
3. Training and evaluation functionality
4. Model persistence

Key components:
- LinearModel class: Core implementation of linear regression for emotion prediction
- Training pipeline with cross-validation
- Evaluation metrics specific to emotion prediction
- Support for both static and dynamic emotion prediction

Notes:
- For static emotions, one model is trained per emotion dimension (e.g., valence, arousal)
- For dynamic emotions, models can predict time-varying emotional qualities
- Ridge regularization helps prevent overfitting with high-dimensional audio features
"""

import numpy as np
import os
import joblib
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LinearModel:
    def __init__(self, use_ridge=True, is_dynamic=False, alpha=1.0):
        """
        Initialize the linear regression model.
        
        Args:
            use_ridge: Whether to use Ridge regression with regularization
            is_dynamic: Whether this model is for dynamic emotion prediction
            alpha: Regularization strength (only used if use_ridge=True)
        """
        self.use_ridge = use_ridge
        self.is_dynamic = is_dynamic
        self.alpha = alpha
        self.models = []  # One model per emotion dimension
        self.emotion_dims = []
        self.scaler = StandardScaler()
        
        logger.info(f"Initialized {'Ridge' if use_ridge else 'Linear'} regression model "
                   f"for {'dynamic' if is_dynamic else 'static'} emotion prediction")
        
    def _create_model(self):
        """
        Create a pipeline with scaling and regression.
        
        Returns:
            sklearn Pipeline with preprocessing and regression model
        """
        if self.use_ridge:
            return Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', Ridge(alpha=self.alpha))
            ])
        else:
            return Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', LinearRegression())
            ])
    
    def train(self, X, y, emotion_dims=None, cv=5):
        """
        Train the model on the given data.
        
        Args:
            X: Feature matrix
            y: Target values (one column per emotion dimension)
            emotion_dims: Names of emotion dimensions
            cv: Number of cross-validation folds
        """
        # Set default emotion dimensions if not provided
        if emotion_dims is None:
            self.emotion_dims = [f"emotion_{i}" for i in range(y.shape[1])]
        else:
            self.emotion_dims = emotion_dims
            
        logger.info(f"Training {'Ridge' if self.use_ridge else 'Linear'} regression model "
                   f"for {len(self.emotion_dims)} emotion dimensions: {self.emotion_dims}")
        
        # Check input data
        if len(X) == 0 or len(y) == 0:
            logger.error("Empty training data provided")
            raise ValueError("Cannot train on empty data")
        
        logger.info(f"Training data: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"Target shape: {y.shape}")
        
        # Split the data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train one model for each emotion dimension
        for i, dim in enumerate(self.emotion_dims):
            logger.info(f"Training for dimension: {dim} ({i+1}/{len(self.emotion_dims)})")
            
            # Create and configure model
            model = self._create_model()
            
            # Define hyperparameters to search
            if self.use_ridge:
                param_grid = {'regressor__alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}
            else:
                # LinearRegression doesn't have hyperparameters to tune
                param_grid = {}
                
            try:
                # If we have parameters to tune, use GridSearchCV
                if param_grid:
                    logger.info(f"Performing grid search for {dim} with {param_grid}")
                    grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='neg_mean_squared_error')
                    grid_search.fit(X_train, y_train[:, i])
                    
                    # Get best model
                    best_model = grid_search.best_estimator_
                    best_params = grid_search.best_params_
                    logger.info(f"Best parameters for {dim}: {best_params}")
                else:
                    # Otherwise train directly
                    best_model = model
                    best_model.fit(X_train, y_train[:, i])
                
                # Evaluate on validation set
                y_pred = best_model.predict(X_val)
                mse = mean_squared_error(y_val[:, i], y_pred)
                r2 = r2_score(y_val[:, i], y_pred)
                
                logger.info(f"Validation metrics for {dim}:")
                logger.info(f"  MSE: {mse:.4f}")
                logger.info(f"  RMSE: {np.sqrt(mse):.4f}")
                logger.info(f"  R²: {r2:.4f}")
                
                # Save the trained model
                self.models.append(best_model)
                
            except Exception as e:
                logger.error(f"Error training model for {dim}: {e}")
                raise
    
    def predict(self, X):
        """
        Make predictions with the trained model.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted emotion values (one column per dimension)
        """
        if not self.models:
            logger.error("Model not trained yet")
            raise ValueError("Model not trained yet")
            
        # Make predictions for each emotion dimension
        predictions = np.zeros((X.shape[0], len(self.models)))
        
        for i, model in enumerate(self.models):
            predictions[:, i] = model.predict(X)
            
        return predictions
    
    def save(self, model_dir="models"):
        """
        Save the trained models to disk.
        
        Args:
            model_dir: Directory to save models
        """
        os.makedirs(model_dir, exist_ok=True)
        
        model_type = "ridge" if self.use_ridge else "linear"
        dynamic_str = "dynamic" if self.is_dynamic else "static"
        
        for i, (model, dim) in enumerate(zip(self.models, self.emotion_dims)):
            # Create a safe filename from dimension name
            safe_dim = dim.lower().replace(' ', '_')
            filename = f"{model_dir}/{model_type}_{dynamic_str}_{safe_dim}.joblib"
            
            try:
                joblib.dump(model, filename)
                logger.info(f"Model saved to {filename}")
            except Exception as e:
                logger.error(f"Error saving model to {filename}: {e}")
    
    def load(self, model_dir="models"):
        """
        Load trained models from disk.
        
        Args:
            model_dir: Directory to load models from
        """
        model_type = "ridge" if self.use_ridge else "linear"
        dynamic_str = "dynamic" if self.is_dynamic else "static"
        
        self.models = []
        
        for dim in self.emotion_dims:
            # Create a safe filename from dimension name
            safe_dim = dim.lower().replace(' ', '_')
            filename = f"{model_dir}/{model_type}_{dynamic_str}_{safe_dim}.joblib"
            
            try:
                logger.info(f"Loading model from {filename}")
                model = joblib.load(filename)
                self.models.append(model)
            except Exception as e:
                logger.error(f"Error loading model from {filename}: {e}")
                raise
