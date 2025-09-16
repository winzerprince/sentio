"""
Support Vector Regression (SVR) Model for Emotion Prediction

This module implements SVR models for predicting emotions from audio features.
The module includes:

1. Support Vector Regression implementation with different kernels
2. Parameter tuning through cross-validation
3. Training and evaluation functionality
4. Model persistence

Key components:
- SVRModel class: Core implementation of SVR for emotion prediction
- Kernel selection: RBF, Linear, Poly options for different data relationships
- Support for both static and dynamic emotion prediction

Notes:
- SVR models can capture nonlinear relationships between audio features and emotions
- Grid search is used to find optimal hyperparameters for each emotion dimension
- RBF kernel is a good default for most audio emotion tasks
- Feature scaling is critical for SVR performance
"""

import numpy as np
import os
import joblib
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SVRModel:
    def __init__(self, kernel='rbf', is_dynamic=False):
        """
        Initialize the SVR model.
        
        Args:
            kernel: Kernel type to use ('rbf', 'linear', or 'poly')
            is_dynamic: Whether this model is for dynamic emotion prediction
        """
        self.kernel = kernel
        self.is_dynamic = is_dynamic
        self.models = []  # One model per emotion dimension
        self.emotion_dims = []
        
        logger.info(f"Initialized SVR model with {kernel} kernel "
                   f"for {'dynamic' if is_dynamic else 'static'} emotion prediction")
        
    def _create_model(self):
        """
        Create a pipeline with scaling and SVR.
        
        Returns:
            sklearn Pipeline with preprocessing and SVR model
        """
        return Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', SVR(kernel=self.kernel))
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
            
        logger.info(f"Training SVR model with {self.kernel} kernel for "
                   f"{len(self.emotion_dims)} emotion dimensions: {self.emotion_dims}")
        
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
            
            # Define hyperparameters to search based on kernel
            param_grid = {'regressor__C': [0.1, 1, 10, 100]}
            
            if self.kernel == 'rbf':
                param_grid['regressor__gamma'] = ['scale', 'auto', 0.01, 0.1, 1]
            elif self.kernel == 'poly':
                param_grid['regressor__degree'] = [2, 3, 4]
                
            try:
                # Perform grid search to find optimal hyperparameters
                logger.info(f"Performing grid search for {dim} with {param_grid}")
                grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='neg_mean_squared_error')
                grid_search.fit(X_train, y_train[:, i])
                
                # Get best model
                best_model = grid_search.best_estimator_
                best_params = grid_search.best_params_
                logger.info(f"Best parameters for {dim}: {best_params}")
                
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
        
        dynamic_str = "dynamic" if self.is_dynamic else "static"
        
        for i, (model, dim) in enumerate(zip(self.models, self.emotion_dims)):
            # Create a safe filename from dimension name
            safe_dim = dim.lower().replace(' ', '_')
            filename = f"{model_dir}/svr_{self.kernel}_{dynamic_str}_{safe_dim}.joblib"
            
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
        dynamic_str = "dynamic" if self.is_dynamic else "static"
        
        self.models = []
        
        for dim in self.emotion_dims:
            # Create a safe filename from dimension name
            safe_dim = dim.lower().replace(' ', '_')
            filename = f"{model_dir}/svr_{self.kernel}_{dynamic_str}_{safe_dim}.joblib"
            
            try:
                logger.info(f"Loading model from {filename}")
                model = joblib.load(filename)
                self.models.append(model)
            except Exception as e:
                logger.error(f"Error loading model from {filename}: {e}")
                raise
