"""
Comprehensive Model Testing Framework for Music Emotion Analysis & Generation
Designed for HP EliteBook 840 G3 (i5-6300U, 16GB RAM, Intel HD Graphics 520)
"""

import pandas as pd
import numpy as np
import os
import time
import json
import pickle
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Machine Learning Libraries
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb

# Visualization Libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Memory and Performance Monitoring
import psutil
import gc

class SystemMonitor:
    """Monitor system resources during model training"""
    
    def __init__(self):
        self.start_time = None
        self.peak_memory = 0
        self.initial_memory = psutil.virtual_memory().used / (1024**3)  # GB
    
    def start_monitoring(self):
        self.start_time = time.time()
        self.peak_memory = psutil.virtual_memory().used / (1024**3)
    
    def get_current_stats(self) -> Dict[str, float]:
        current_memory = psutil.virtual_memory().used / (1024**3)
        self.peak_memory = max(self.peak_memory, current_memory)
        
        return {
            'current_memory_gb': current_memory,
            'peak_memory_gb': self.peak_memory,
            'memory_increase_gb': current_memory - self.initial_memory,
            'elapsed_time_minutes': (time.time() - self.start_time) / 60 if self.start_time else 0,
            'cpu_percent': psutil.cpu_percent(interval=1)
        }

class DatasetManager:
    """Manage DEAM dataset loading and preprocessing optimized for limited resources"""
    
    def __init__(self, base_path: str, max_samples: int = 1400):
        self.base_path = base_path
        self.max_samples = max_samples
        self.features_df = None
        self.annotations_df = None
        
    def load_annotations(self) -> pd.DataFrame:
        """Load emotion annotations (valence/arousal)"""
        ann_path1 = os.path.join(
            self.base_path, 
            "annotations/annotations averaged per song/song_level/static_annotations_averaged_songs_1_2000.csv"
        )
        ann_path2 = os.path.join(
            self.base_path, 
            "annotations/annotations averaged per song/song_level/static_annotations_averaged_songs_2000_2058.csv"
        )
        
        df1 = pd.read_csv(ann_path1, skipinitialspace=True)
        df2 = pd.read_csv(ann_path2, skipinitialspace=True)
        
        self.annotations_df = pd.concat([df1, df2], ignore_index=True)
        
        # Normalize to 0-1 scale (originally 1-9)
        self.annotations_df['valence_norm'] = (self.annotations_df['valence_mean'] - 1) / 8
        self.annotations_df['arousal_norm'] = (self.annotations_df['arousal_mean'] - 1) / 8
        
        print(f"Loaded {len(self.annotations_df)} song annotations")
        return self.annotations_df
    
    def load_features(self, sample_size: Optional[int] = None) -> pd.DataFrame:
        """Load audio features with memory optimization"""
        if sample_size is None:
            sample_size = self.max_samples
            
        features_dir = os.path.join(self.base_path, "features")
        available_files = [f for f in os.listdir(features_dir) if f.endswith('.csv')]
        
        # Get song IDs that have both features and annotations
        song_ids = []
        for file in available_files:
            song_id = int(file.replace('.csv', ''))
            if song_id in self.annotations_df['song_id'].values:
                song_ids.append(song_id)
        
        # Limit to specified sample size
        song_ids = sorted(song_ids)[:sample_size]
        print(f"Loading features for {len(song_ids)} songs (limited by sample_size={sample_size})")
        
        # Load features in batches to manage memory
        feature_data = []
        batch_size = 100
        
        for i in range(0, len(song_ids), batch_size):
            batch_ids = song_ids[i:i+batch_size]
            batch_features = []
            
            for song_id in batch_ids:
                file_path = os.path.join(features_dir, f"{song_id}.csv")
                try:
                    df = pd.read_csv(file_path, delimiter=';')
                    # Aggregate features (mean and std) to get song-level representation
                    feature_vector = []
                    for col in df.columns[1:]:  # Skip frameTime
                        if df[col].dtype in ['float64', 'int64']:
                            feature_vector.extend([
                                df[col].mean(),
                                df[col].std()
                            ])
                    
                    feature_row = {'song_id': song_id}
                    for j, val in enumerate(feature_vector):
                        feature_row[f'feature_{j}'] = val
                    
                    batch_features.append(feature_row)
                    
                except Exception as e:
                    print(f"Error loading {song_id}.csv: {e}")
                    continue
            
            feature_data.extend(batch_features)
            
            # Memory cleanup
            gc.collect()
            
            if i % 500 == 0:
                current_mem = psutil.virtual_memory().used / (1024**3)
                print(f"Processed {i+len(batch_ids)} files, Memory usage: {current_mem:.2f}GB")
        
        self.features_df = pd.DataFrame(feature_data)
        
        # Handle missing values
        self.features_df = self.features_df.fillna(0)
        
        print(f"Final feature dataset shape: {self.features_df.shape}")
        return self.features_df
    
    def get_merged_dataset(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Merge features with annotations"""
        merged = self.features_df.merge(
            self.annotations_df[['song_id', 'valence_norm', 'arousal_norm', 'valence_mean', 'arousal_mean']], 
            on='song_id'
        )
        
        feature_cols = [col for col in merged.columns if col.startswith('feature_')]
        X = merged[feature_cols]
        y = merged[['valence_norm', 'arousal_norm']]
        
        print(f"Merged dataset: X shape {X.shape}, y shape {y.shape}")
        return X, y

class ModelEvaluator:
    """Comprehensive model evaluation with multiple metrics"""
    
    @staticmethod
    def evaluate_regression_model(y_true: np.ndarray, y_pred: np.ndarray, 
                                model_name: str) -> Dict[str, float]:
        """Calculate comprehensive regression metrics"""
        metrics = {}
        
        if y_true.ndim > 1:  # Multi-output (valence + arousal)
            for i, target in enumerate(['valence', 'arousal']):
                metrics[f'{target}_mse'] = mean_squared_error(y_true[:, i], y_pred[:, i])
                metrics[f'{target}_rmse'] = np.sqrt(metrics[f'{target}_mse'])
                metrics[f'{target}_mae'] = mean_absolute_error(y_true[:, i], y_pred[:, i])
                metrics[f'{target}_r2'] = r2_score(y_true[:, i], y_pred[:, i])
                
            # Overall metrics
            metrics['overall_mse'] = mean_squared_error(y_true, y_pred)
            metrics['overall_rmse'] = np.sqrt(metrics['overall_mse'])
            metrics['overall_mae'] = mean_absolute_error(y_true, y_pred)
            
        else:  # Single output
            metrics['mse'] = mean_squared_error(y_true, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
            metrics['r2'] = r2_score(y_true, y_pred)
        
        return metrics

class AnalysisModelTester:
    """Test emotion analysis models with system constraints in mind"""
    
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.results = {}
        self.monitor = SystemMonitor()
        
        # Preprocess data
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(X_train)
        self.X_test_scaled = self.scaler.transform(X_test)
        
    def test_ridge_regression(self) -> Dict[str, Any]:
        """Test Ridge Regression (Lightweight)"""
        print("\n=== Testing Ridge Regression ===")
        self.monitor.start_monitoring()
        
        # Grid search with conservative parameters for limited resources
        param_grid = {
            'alpha': [0.1, 1.0, 10.0, 100.0]
        }
        
        results = {}
        
        # Test for valence and arousal separately (less memory intensive)
        for i, target in enumerate(['valence', 'arousal']):
            print(f"Training {target} predictor...")
            
            ridge = Ridge()
            grid_search = GridSearchCV(
                ridge, param_grid, cv=3, 
                scoring='neg_mean_squared_error',
                n_jobs=2  # Use half the cores
            )
            
            grid_search.fit(self.X_train_scaled, self.y_train.iloc[:, i])
            best_model = grid_search.best_estimator_
            
            # Predictions
            y_pred = best_model.predict(self.X_test_scaled)
            
            # Evaluate
            metrics = ModelEvaluator.evaluate_regression_model(
                self.y_test.iloc[:, i].values, y_pred, f'ridge_{target}'
            )
            
            results[target] = {
                'model': best_model,
                'best_params': grid_search.best_params_,
                'metrics': metrics,
                'predictions': y_pred
            }
        
        results['system_stats'] = self.monitor.get_current_stats()
        self.results['ridge'] = results
        
        print(f"Ridge Regression completed - Memory: {results['system_stats']['peak_memory_gb']:.2f}GB, "
              f"Time: {results['system_stats']['elapsed_time_minutes']:.2f}min")
        
        return results
    
    def test_svr(self) -> Dict[str, Any]:
        """Test Support Vector Regression (Lightweight)"""
        print("\n=== Testing Support Vector Regression ===")
        self.monitor.start_monitoring()
        
        # Conservative parameters for limited resources
        param_grid = {
            'kernel': ['rbf'],  # Only RBF to save time
            'C': [1, 10],       # Reduced parameter space
            'gamma': ['scale', 'auto']
        }
        
        results = {}
        
        for i, target in enumerate(['valence', 'arousal']):
            print(f"Training {target} predictor...")
            
            svr = SVR()
            
            # Use smaller subset for grid search to save time
            subset_size = min(800, len(self.X_train_scaled))
            indices = np.random.choice(len(self.X_train_scaled), subset_size, replace=False)
            
            grid_search = GridSearchCV(
                svr, param_grid, cv=3,
                scoring='neg_mean_squared_error',
                n_jobs=2
            )
            
            grid_search.fit(
                self.X_train_scaled[indices], 
                self.y_train.iloc[indices, i]
            )
            
            # Train best model on full training set
            best_model = grid_search.best_estimator_
            best_model.fit(self.X_train_scaled, self.y_train.iloc[:, i])
            
            # Predictions
            y_pred = best_model.predict(self.X_test_scaled)
            
            # Evaluate
            metrics = ModelEvaluator.evaluate_regression_model(
                self.y_test.iloc[:, i].values, y_pred, f'svr_{target}'
            )
            
            results[target] = {
                'model': best_model,
                'best_params': grid_search.best_params_,
                'metrics': metrics,
                'predictions': y_pred
            }
        
        results['system_stats'] = self.monitor.get_current_stats()
        self.results['svr'] = results
        
        print(f"SVR completed - Memory: {results['system_stats']['peak_memory_gb']:.2f}GB, "
              f"Time: {results['system_stats']['elapsed_time_minutes']:.2f}min")
        
        return results
    
    def test_xgboost(self) -> Dict[str, Any]:
        """Test XGBoost (Heavyweight)"""
        print("\n=== Testing XGBoost ===")
        self.monitor.start_monitoring()
        
        # Conservative parameters for limited resources
        param_grid = {
            'n_estimators': [50, 100],      # Reduced from typical 100-1000
            'max_depth': [3, 6],            # Shallow trees
            'learning_rate': [0.1, 0.2],    # Faster learning
            'subsample': [0.8],             # Reduce memory usage
            'colsample_bytree': [0.8]       # Feature sampling
        }
        
        results = {}
        
        for i, target in enumerate(['valence', 'arousal']):
            print(f"Training {target} predictor...")
            
            xgb_model = xgb.XGBRegressor(
                random_state=42,
                n_jobs=2,               # Use 2 cores
                tree_method='hist'      # Memory efficient
            )
            
            # Use smaller subset for grid search
            subset_size = min(600, len(self.X_train_scaled))
            indices = np.random.choice(len(self.X_train_scaled), subset_size, replace=False)
            
            grid_search = GridSearchCV(
                xgb_model, param_grid, cv=3,
                scoring='neg_mean_squared_error',
                n_jobs=1  # Sequential to avoid memory issues
            )
            
            grid_search.fit(
                self.X_train_scaled[indices], 
                self.y_train.iloc[indices, i]
            )
            
            # Train best model on full training set
            best_model = grid_search.best_estimator_
            best_model.fit(self.X_train_scaled, self.y_train.iloc[:, i])
            
            # Predictions
            y_pred = best_model.predict(self.X_test_scaled)
            
            # Evaluate
            metrics = ModelEvaluator.evaluate_regression_model(
                self.y_test.iloc[:, i].values, y_pred, f'xgb_{target}'
            )
            
            # Feature importance
            feature_importance = best_model.feature_importances_
            
            results[target] = {
                'model': best_model,
                'best_params': grid_search.best_params_,
                'metrics': metrics,
                'predictions': y_pred,
                'feature_importance': feature_importance
            }
            
            # Memory cleanup
            gc.collect()
        
        results['system_stats'] = self.monitor.get_current_stats()
        self.results['xgboost'] = results
        
        print(f"XGBoost completed - Memory: {results['system_stats']['peak_memory_gb']:.2f}GB, "
              f"Time: {results['system_stats']['elapsed_time_minutes']:.2f}min")
        
        return results
    
    def test_mlp(self) -> Dict[str, Any]:
        """Test Multi-layer Perceptron (Heavyweight)"""
        print("\n=== Testing Multi-layer Perceptron ===")
        self.monitor.start_monitoring()
        
        # Conservative parameters for limited resources
        param_grid = {
            'hidden_layer_sizes': [(32,), (64,), (32, 16)],  # Small networks
            'alpha': [0.01, 0.1],                           # Regularization
            'learning_rate_init': [0.01, 0.001],           # Learning rate
            'max_iter': [200]                               # Reduced iterations
        }
        
        results = {}
        
        for i, target in enumerate(['valence', 'arousal']):
            print(f"Training {target} predictor...")
            
            mlp = MLPRegressor(
                random_state=42,
                early_stopping=True,    # Stop early if no improvement
                validation_fraction=0.1,
                n_iter_no_change=10
            )
            
            # Use smaller subset for grid search
            subset_size = min(500, len(self.X_train_scaled))
            indices = np.random.choice(len(self.X_train_scaled), subset_size, replace=False)
            
            grid_search = GridSearchCV(
                mlp, param_grid, cv=3,
                scoring='neg_mean_squared_error',
                n_jobs=1  # Sequential to avoid memory issues
            )
            
            grid_search.fit(
                self.X_train_scaled[indices], 
                self.y_train.iloc[indices, i]
            )
            
            # Train best model on full training set
            best_model = grid_search.best_estimator_
            best_model.fit(self.X_train_scaled, self.y_train.iloc[:, i])
            
            # Predictions
            y_pred = best_model.predict(self.X_test_scaled)
            
            # Evaluate
            metrics = ModelEvaluator.evaluate_regression_model(
                self.y_test.iloc[:, i].values, y_pred, f'mlp_{target}'
            )
            
            results[target] = {
                'model': best_model,
                'best_params': grid_search.best_params_,
                'metrics': metrics,
                'predictions': y_pred
            }
            
            # Memory cleanup
            gc.collect()
        
        results['system_stats'] = self.monitor.get_current_stats()
        self.results['mlp'] = results
        
        print(f"MLP completed - Memory: {results['system_stats']['peak_memory_gb']:.2f}GB, "
              f"Time: {results['system_stats']['elapsed_time_minutes']:.2f}min")
        
        return results
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all analysis model tests"""
        print("Starting comprehensive analysis model testing...")
        print(f"Dataset size: Train={len(self.X_train)}, Test={len(self.X_test)}")
        print(f"Feature dimensions: {self.X_train.shape[1]}")
        
        # Test lightweight models
        self.test_ridge_regression()
        self.test_svr()
        
        # Test heavyweight models (with memory monitoring)
        current_mem = psutil.virtual_memory().used / (1024**3)
        if current_mem < 12:  # Only if we have enough memory
            self.test_xgboost()
            self.test_mlp()
        else:
            print("Skipping heavyweight models due to memory constraints")
        
        return self.results

class ReportGenerator:
    """Generate comprehensive performance reports"""
    
    def __init__(self, results: Dict[str, Any]):
        self.results = results
    
    def generate_performance_comparison(self) -> pd.DataFrame:
        """Generate performance comparison table"""
        comparison_data = []
        
        for model_name, model_results in self.results.items():
            if 'system_stats' not in model_results:
                continue
                
            for target in ['valence', 'arousal']:
                if target in model_results:
                    metrics = model_results[target]['metrics']
                    system_stats = model_results['system_stats']
                    
                    comparison_data.append({
                        'Model': model_name.upper(),
                        'Target': target.capitalize(),
                        'RMSE': metrics.get('rmse', metrics.get(f'{target}_rmse', 'N/A')),
                        'MAE': metrics.get('mae', metrics.get(f'{target}_mae', 'N/A')),
                        'R²': metrics.get('r2', metrics.get(f'{target}_r2', 'N/A')),
                        'Training_Time_min': system_stats['elapsed_time_minutes'],
                        'Peak_Memory_GB': system_stats['peak_memory_gb'],
                        'Memory_Increase_GB': system_stats['memory_increase_gb']
                    })
        
        return pd.DataFrame(comparison_data)
    
    def create_visualizations(self, save_dir: str):
        """Create performance visualization plots"""
        os.makedirs(save_dir, exist_ok=True)
        
        comparison_df = self.generate_performance_comparison()
        
        # Performance comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # RMSE comparison
        rmse_data = comparison_df.pivot(index='Model', columns='Target', values='RMSE')
        rmse_data.plot(kind='bar', ax=axes[0,0], title='RMSE Comparison')
        axes[0,0].set_ylabel('RMSE')
        axes[0,0].legend(title='Target')
        
        # R² comparison
        r2_data = comparison_df.pivot(index='Model', columns='Target', values='R²')
        r2_data.plot(kind='bar', ax=axes[0,1], title='R² Score Comparison')
        axes[0,1].set_ylabel('R² Score')
        axes[0,1].legend(title='Target')
        
        # Training time comparison
        time_data = comparison_df.groupby('Model')['Training_Time_min'].first()
        time_data.plot(kind='bar', ax=axes[1,0], title='Training Time Comparison')
        axes[1,0].set_ylabel('Training Time (minutes)')
        
        # Memory usage comparison
        memory_data = comparison_df.groupby('Model')['Peak_Memory_GB'].first()
        memory_data.plot(kind='bar', ax=axes[1,1], title='Peak Memory Usage Comparison')
        axes[1,1].set_ylabel('Peak Memory (GB)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        return comparison_df
    
    def save_detailed_report(self, save_path: str):
        """Save detailed report to JSON"""
        with open(save_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)

def main():
    """Main testing pipeline"""
    # Configuration
    BASE_PATH = "/mnt/sdb8mount/free-explore/class/ai/datasets/sentio/dataset/DEAM"
    RESULTS_DIR = "/mnt/sdb8mount/free-explore/class/ai/datasets/sentio/results"
    SAMPLE_SIZE = 1200  # Adjusted for system constraints
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print("=" * 80)
    print("MUSIC EMOTION ANALYSIS - COMPREHENSIVE MODEL TESTING")
    print("=" * 80)
    print(f"System: HP EliteBook 840 G3 (i5-6300U, 16GB RAM)")
    print(f"Sample size: {SAMPLE_SIZE} songs")
    print(f"Results directory: {RESULTS_DIR}")
    print("=" * 80)
    
    # Load and prepare data
    print("\n1. Loading dataset...")
    dataset_manager = DatasetManager(BASE_PATH, SAMPLE_SIZE)
    
    annotations = dataset_manager.load_annotations()
    features = dataset_manager.load_features(SAMPLE_SIZE)
    X, y = dataset_manager.get_merged_dataset()
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=None
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Run model tests
    print("\n2. Running model tests...")
    tester = AnalysisModelTester(X_train, X_test, y_train, y_test)
    results = tester.run_all_tests()
    
    # Generate reports
    print("\n3. Generating reports...")
    report_generator = ReportGenerator(results)
    
    # Save comparison table
    comparison_df = report_generator.create_visualizations(RESULTS_DIR)
    comparison_df.to_csv(os.path.join(RESULTS_DIR, 'model_comparison.csv'), index=False)
    
    # Save detailed results
    report_generator.save_detailed_report(os.path.join(RESULTS_DIR, 'detailed_results.json'))
    
    # Print summary
    print("\n" + "=" * 80)
    print("TESTING COMPLETED - SUMMARY")
    print("=" * 80)
    print(comparison_df.to_string(index=False))
    print("=" * 80)
    print(f"Detailed results saved to: {RESULTS_DIR}")
    
if __name__ == "__main__":
    main()
