"""
XGBoost Regression Model for Emotion Prediction

This module implements XGBoost regression models for predicting emotions from audio features.
The module includes:

1. XGBoost regression implementation for high performance
2. Hyperparameter tuning through cross-validation
3. Training and evaluation functionality
4. Model persistence

Key components:
- XGBoostModel class: Core implementation of XGBoost for emotion prediction
- Feature importance analysis
- Early stopping to prevent overfitting
- Support for both static and dynamic emotion prediction

Notes:
- XGBoost can capture complex nonlinear relationships in audio feature data
- It often outperforms linear models and sometimes SVR for emotion prediction
- Grid search is used to find optimal hyperparameters for each emotion dimension
- Feature importance can provide insights into which audio features best predict emotions
"""

import numpy as np
import os
import joblib
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import logging
import matplotlib.pyplot as plt
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class XGBoostModel:
    def __init__(self, is_dynamic=False):
        """
        Initialize the XGBoost model.
        
        Args:
            is_dynamic: Whether this model is for dynamic emotion prediction
        """
        self.is_dynamic = is_dynamic
        self.models = []  # One model per emotion dimension
        self.emotion_dims = []
        self.feature_importances = {}  # To store feature importance for each dimension
        
        logger.info(f"Initialized XGBoost model for "
                   f"{'dynamic' if is_dynamic else 'static'} emotion prediction")
        
    def _create_model(self):
        """
        Create a pipeline with scaling and XGBoost.
        
        Returns:
            sklearn Pipeline with preprocessing and XGBoost model
        """
        return Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', xgb.XGBRegressor(
                objective='reg:squarederror',  # For regression tasks
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            ))
        ])
    
    def train(self, X, y, emotion_dims=None, feature_names=None, cv=5):
        """
        Train the model on the given data.
        
        Args:
            X: Feature matrix
            y: Target values (one column per emotion dimension)
            emotion_dims: Names of emotion dimensions
            feature_names: Names of input features (for feature importance)
            cv: Number of cross-validation folds
        """
        # Set default emotion dimensions if not provided
        if emotion_dims is None:
            self.emotion_dims = [f"emotion_{i}" for i in range(y.shape[1])]
        else:
            self.emotion_dims = emotion_dims
            
        # Set feature names if provided
        self.feature_names = feature_names
        if feature_names is None:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        logger.info(f"Training XGBoost model for "
                   f"{len(self.emotion_dims)} emotion dimensions: {self.emotion_dims}")
        
        # Check input data
        if len(X) == 0 or len(y) == 0:
            logger.error("Empty training data provided")
            raise ValueError("Cannot train on empty data")
        
        logger.info(f"Training data: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"Target shape: {y.shape}")
        
        # Split the data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create evaluation set for early stopping
        eval_set = [(X_val, None)]  # Will be updated for each dimension
        
        # Train one model for each emotion dimension
        for i, dim in enumerate(self.emotion_dims):
            logger.info(f"Training for dimension: {dim} ({i+1}/{len(self.emotion_dims)})")
            
            # Create and configure model
            model = self._create_model()
            
            # Define hyperparameters to search
            param_grid = {
                'regressor__n_estimators': [50, 100, 200],
                'regressor__learning_rate': [0.01, 0.05, 0.1],
                'regressor__max_depth': [3, 5, 7]
            }
                
            try:
                # Perform grid search to find optimal hyperparameters
                logger.info(f"Performing grid search for {dim}")
                grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='neg_mean_squared_error')
                grid_search.fit(X_train, y_train[:, i])
                
                # Get best model
                best_model = grid_search.best_estimator_
                best_params = grid_search.best_params_
                logger.info(f"Best parameters for {dim}: {best_params}")
                
                # Update the regressor with early stopping
                best_regressor = best_model.named_steps['regressor']
                best_regressor.set_params(early_stopping_rounds=10)
                
                # Retrain with early stopping
                # We need to extract the original XGBoost model and train it directly 
                # to use early_stopping_rounds
                xgb_model = xgb.XGBRegressor(
                    objective='reg:squarederror',
                    n_estimators=best_params.get('regressor__n_estimators', 100),
                    learning_rate=best_params.get('regressor__learning_rate', 0.1),
                    max_depth=best_params.get('regressor__max_depth', 5),
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    early_stopping_rounds=10
                )
                
                # Scale features
                scaler = best_model.named_steps['scaler']
                X_train_scaled = scaler.transform(X_train)
                X_val_scaled = scaler.transform(X_val)
                
                # Train with early stopping
                eval_set = [(X_train_scaled, y_train[:, i]), (X_val_scaled, y_val[:, i])]
                xgb_model.fit(X_train_scaled, y_train[:, i], 
                             eval_set=eval_set,
                             verbose=False)
                
                # Recreate pipeline with trained components
                final_model = Pipeline([
                    ('scaler', scaler),
                    ('regressor', xgb_model)
                ])
                
                # Evaluate on validation set
                y_pred = final_model.predict(X_val)
                mse = mean_squared_error(y_val[:, i], y_pred)
                r2 = r2_score(y_val[:, i], y_pred)
                
                logger.info(f"Validation metrics for {dim}:")
                logger.info(f"  MSE: {mse:.4f}")
                logger.info(f"  RMSE: {np.sqrt(mse):.4f}")
                logger.info(f"  R²: {r2:.4f}")
                
                # Get feature importances
                importances = xgb_model.feature_importances_
                self.feature_importances[dim] = {
                    'values': importances,
                    'names': self.feature_names
                }
                
                # Save the trained model
                self.models.append(final_model)
                
                # Plot feature importance
                self._plot_feature_importance(dim, importances)
                
            except Exception as e:
                logger.error(f"Error training model for {dim}: {e}")
                raise
    
    def _plot_feature_importance(self, dimension, importances, top_n=20):
        """
        Plot feature importance for a specific emotion dimension.
        
        Args:
            dimension: Emotion dimension name
            importances: Array of feature importance values
            top_n: Number of top features to display
        """
        try:
            # Create directory for plots
            os.makedirs('plots', exist_ok=True)
            
            # Get the top N important features
            indices = np.argsort(importances)[::-1][:top_n]
            if self.feature_names and len(self.feature_names) > 0:
                top_features = [self.feature_names[i] if i < len(self.feature_names) else f"feature_{i}" 
                              for i in indices]
            else:
                top_features = [f"feature_{i}" for i in indices]
            top_importances = importances[indices]
            
            # Create plot
            plt.figure(figsize=(10, 8))
            plt.title(f'Top {top_n} Feature Importance for {dimension}')
            plt.barh(range(top_n), top_importances, align='center')
            plt.yticks(range(top_n), top_features)
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.tight_layout()
            
            # Save plot
            safe_dim = dimension.lower().replace(' ', '_')
            plt.savefig(f'plots/xgboost_importance_{safe_dim}.png')
            plt.close()
            
            logger.info(f"Feature importance plot saved for {dimension}")
            
            # Also save as CSV for further analysis
            feature_names_list = self.feature_names if self.feature_names else [f"feature_{i}" for i in range(len(importances))]
            importance_df = pd.DataFrame({
                'Feature': feature_names_list,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            importance_df.to_csv(f'plots/xgboost_importance_{safe_dim}.csv', index=False)
            
        except Exception as e:
            logger.warning(f"Could not plot feature importance: {e}")
    
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
            filename = f"{model_dir}/xgboost_{dynamic_str}_{safe_dim}.joblib"
            
            try:
                joblib.dump(model, filename)
                logger.info(f"Model saved to {filename}")
            except Exception as e:
                logger.error(f"Error saving model to {filename}: {e}")
                
        # Save feature importances
        if self.feature_importances:
            try:
                os.makedirs(os.path.join(model_dir, 'feature_importance'), exist_ok=True)
                for dim, importance in self.feature_importances.items():
                    safe_dim = dim.lower().replace(' ', '_')
                    filename = f"{model_dir}/feature_importance/xgboost_{dynamic_str}_{safe_dim}.joblib"
                    joblib.dump(importance, filename)
                    logger.info(f"Feature importance saved to {filename}")
            except Exception as e:
                logger.error(f"Error saving feature importance: {e}")
    
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
            filename = f"{model_dir}/xgboost_{dynamic_str}_{safe_dim}.joblib"
            
            try:
                logger.info(f"Loading model from {filename}")
                model = joblib.load(filename)
                self.models.append(model)
            except Exception as e:
                logger.error(f"Error loading model from {filename}: {e}")
                raise
                
        # Load feature importances
        self.feature_importances = {}
        for dim in self.emotion_dims:
            safe_dim = dim.lower().replace(' ', '_')
            filename = f"{model_dir}/feature_importance/xgboost_{dynamic_str}_{safe_dim}.joblib"
            try:
                self.feature_importances[dim] = joblib.load(filename)
            except Exception as e:
                logger.debug(f"Could not load feature importance for {dim}: {e}")
                # This is not critical, so we can continue
