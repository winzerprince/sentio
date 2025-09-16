"""
Model Comparison Prediction Script

This script compares predictions from SVR and XGBoost models side by side
to understand their differences in predicting valence and arousal.
"""
import os
import joblib
import pandas as pd
import numpy as np
import logging
import random

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

def compare_models(sample_files, models_dir='output/models'):
    """
    Compare SVR and XGBoost predictions on multiple samples.
    """
    try:
        # Load the scaler
        scaler_path = os.path.join(models_dir, 'scaler.pkl')
        scaler = joblib.load(scaler_path)
        logger.info(f"Scaler loaded from {scaler_path}")

        # Load SVR models
        svr_valence_path = os.path.join(models_dir, 'svr', 'svr_rbf_static_valence.joblib')
        svr_arousal_path = os.path.join(models_dir, 'svr', 'svr_rbf_static_arousal.joblib')
        
        svr_valence_model = joblib.load(svr_valence_path)
        svr_arousal_model = joblib.load(svr_arousal_path)
        logger.info("SVR models loaded.")

        # Load XGBoost models
        xgb_valence_path = os.path.join(models_dir, 'xgboost', 'xgboost_static_valence.joblib')
        xgb_arousal_path = os.path.join(models_dir, 'xgboost', 'xgboost_static_arousal.joblib')
        
        xgb_valence_model = joblib.load(xgb_valence_path)
        xgb_arousal_model = joblib.load(xgb_arousal_path)
        logger.info("XGBoost models loaded.")

        print("\n" + "="*80)
        print("MODEL COMPARISON: SVR vs XGBoost")
        print("="*80)
        print(f"{'Sample':<15} {'SVR Valence':<12} {'XGB Valence':<12} {'SVR Arousal':<12} {'XGB Arousal':<12} {'Best Model':<15}")
        print("-"*80)

        results = []
        
        for sample_file in sample_files:
            if not os.path.exists(sample_file):
                logger.warning(f"Sample file not found: {sample_file}")
                continue

            # Load and process features
            feature_df = pd.read_csv(sample_file)
            feature_vector = process_static_features(feature_df)
            scaled_features = scaler.transform(feature_vector)

            # Get predictions from both models
            svr_valence = svr_valence_model.predict(scaled_features)[0]
            svr_arousal = svr_arousal_model.predict(scaled_features)[0]
            
            xgb_valence = xgb_valence_model.predict(scaled_features)[0]
            xgb_arousal = xgb_arousal_model.predict(scaled_features)[0]

            # Determine which model's predictions are more "confident" (further from neutral=5)
            svr_confidence = abs(svr_valence - 5) + abs(svr_arousal - 5)
            xgb_confidence = abs(xgb_valence - 5) + abs(xgb_arousal - 5)
            
            best_model = "SVR" if svr_confidence > xgb_confidence else "XGBoost"
            
            # Extract sample name
            sample_name = os.path.basename(sample_file).replace('_selected.csv', '')
            
            print(f"{sample_name:<15} {svr_valence:<12.3f} {xgb_valence:<12.3f} {svr_arousal:<12.3f} {xgb_arousal:<12.3f} {best_model:<15}")
            
            results.append({
                'sample': sample_name,
                'svr_valence': svr_valence,
                'svr_arousal': svr_arousal,
                'xgb_valence': xgb_valence,
                'xgb_arousal': xgb_arousal,
                'valence_diff': abs(svr_valence - xgb_valence),
                'arousal_diff': abs(svr_arousal - xgb_arousal)
            })

        print("-"*80)
        
        # Calculate summary statistics
        valence_diffs = [r['valence_diff'] for r in results]
        arousal_diffs = [r['arousal_diff'] for r in results]
        
        print(f"\nSUMMARY STATISTICS:")
        print(f"Average difference in Valence predictions: {np.mean(valence_diffs):.3f}")
        print(f"Average difference in Arousal predictions: {np.mean(arousal_diffs):.3f}")
        print(f"Max difference in Valence: {np.max(valence_diffs):.3f}")
        print(f"Max difference in Arousal: {np.max(arousal_diffs):.3f}")
        
        # Show scaled interpretations for a few samples
        print(f"\nDETAILED ANALYSIS (First 3 samples):")
        print("="*80)
        
        for i, result in enumerate(results[:3]):
            sample = result['sample']
            print(f"\nSample {sample}:")
            
            # SVR predictions (scaled)
            svr_val_scaled = (result['svr_valence'] - 5.0) / 4.0
            svr_aro_scaled = (result['svr_arousal'] - 5.0) / 4.0
            
            # XGBoost predictions (scaled)
            xgb_val_scaled = (result['xgb_valence'] - 5.0) / 4.0
            xgb_aro_scaled = (result['xgb_arousal'] - 5.0) / 4.0
            
            print(f"  SVR:     Valence={result['svr_valence']:.3f} ({svr_val_scaled:+.3f}), Arousal={result['svr_arousal']:.3f} ({svr_aro_scaled:+.3f})")
            print(f"  XGBoost: Valence={result['xgb_valence']:.3f} ({xgb_val_scaled:+.3f}), Arousal={result['xgb_arousal']:.3f} ({xgb_aro_scaled:+.3f})")
            
            # Interpretations
            svr_emotion = "positive" if svr_val_scaled > 0.1 else "negative" if svr_val_scaled < -0.1 else "neutral"
            svr_energy = "high" if svr_aro_scaled > 0.1 else "low" if svr_aro_scaled < -0.1 else "moderate"
            
            xgb_emotion = "positive" if xgb_val_scaled > 0.1 else "negative" if xgb_val_scaled < -0.1 else "neutral"
            xgb_energy = "high" if xgb_aro_scaled > 0.1 else "low" if xgb_aro_scaled < -0.1 else "moderate"
            
            print(f"  SVR interpretation:     {svr_emotion} emotion, {svr_energy} energy")
            print(f"  XGBoost interpretation: {xgb_emotion} emotion, {xgb_energy} energy")
            
            if svr_emotion != xgb_emotion or svr_energy != xgb_energy:
                print(f"  >>> MODELS DISAGREE <<<")

        return results

    except Exception as e:
        logger.error(f"An error occurred during model comparison: {e}")
        return None

if __name__ == '__main__':
    # Get a random sample of feature files for testing
    import glob
    
    all_files = glob.glob('selected/*_selected.csv')
    
    if len(all_files) < 10:
        sample_files = all_files
    else:
        # Select 10 random samples
        sample_files = random.sample(all_files, 10)
    
    print(f"Testing with {len(sample_files)} sample files...")
    
    results = compare_models(sample_files)
    
    if results:
        print(f"\n✅ Comparison completed successfully!")
        print(f"Note: SVR achieved R²=0.567 for arousal vs XGBoost R²=0.562")
        print(f"This analysis shows how their predictions differ in practice.")
