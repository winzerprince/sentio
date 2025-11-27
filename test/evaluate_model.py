#!/usr/bin/env python3
"""
Evaluate Model Against DEAM Annotations

This script compares model predictions with ground truth DEAM annotations
and visualizes the results with various plots.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from predict import EmotionPredictor


def load_deam_annotations(annotation_file: str) -> pd.DataFrame:
    """
    Load DEAM ground truth annotations.
    
    Args:
        annotation_file: Path to the annotation CSV file
        
    Returns:
        DataFrame with song_id, valence_mean, arousal_mean
    """
    print(f"📖 Loading annotations from: {annotation_file}")
    
    # Read the annotation file
    df = pd.read_csv(annotation_file, skipinitialspace=True)
    
    # Clean column names (remove spaces)
    df.columns = df.columns.str.strip()
    
    print(f"   Found {len(df)} annotated songs")
    print(f"   Columns: {list(df.columns)}")
    
    return df


def normalize_deam_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize DEAM scores from 1-9 scale to -1 to +1 scale.
    
    DEAM uses a 1-9 scale where:
    - 1 = most negative/calm, 9 = most positive/excited
    - Middle point is 5
    
    We convert to -1 to +1 where:
    - -1 = most negative/calm (DEAM score 1)
    - 0 = neutral (DEAM score 5)
    - +1 = most positive/excited (DEAM score 9)
    """
    df = df.copy()
    
    # Convert from 1-9 scale to -1 to +1 scale
    # Formula: (score - 5) / 4
    df['valence_normalized'] = (df['valence_mean'] - 5) / 4
    df['arousal_normalized'] = (df['arousal_mean'] - 5) / 4
    
    # Clip to ensure values are in range [-1, 1]
    df['valence_normalized'] = df['valence_normalized'].clip(-1, 1)
    df['arousal_normalized'] = df['arousal_normalized'].clip(-1, 1)
    
    return df


def predict_emotions(audio_dir: str, song_ids: List[int], 
                     model_path: str, device: str = 'cpu') -> pd.DataFrame:
    """
    Predict emotions for a list of songs.
    
    Args:
        audio_dir: Directory containing audio files
        song_ids: List of song IDs to predict
        model_path: Path to model checkpoint
        device: Device to run inference on
        
    Returns:
        DataFrame with predictions
    """
    print(f"\n🔄 Loading model from: {model_path}")
    predictor = EmotionPredictor(model_path=model_path, device=device)
    
    results = []
    print(f"\n🎵 Predicting emotions for {len(song_ids)} songs...")
    print("=" * 60)
    
    for i, song_id in enumerate(song_ids, 1):
        audio_file = os.path.join(audio_dir, f"{song_id}.mp3")
        
        if not os.path.exists(audio_file):
            print(f"⚠️  [{i}/{len(song_ids)}] Skipping {song_id}.mp3 (file not found)")
            continue
        
        try:
            prediction = predictor.predict(audio_file)
            results.append({
                'song_id': song_id,
                'predicted_valence': prediction['valence'],
                'predicted_arousal': prediction['arousal']
            })
            print(f"✓  [{i}/{len(song_ids)}] {song_id}.mp3")
            
        except Exception as e:
            print(f"❌ [{i}/{len(song_ids)}] Error processing {song_id}.mp3: {e}")
    
    print("=" * 60)
    return pd.DataFrame(results)


def calculate_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate evaluation metrics.
    
    Args:
        df: DataFrame with ground_truth and predictions
        
    Returns:
        Dictionary of metrics
    """
    # Mean Absolute Error (MAE)
    mae_valence = np.mean(np.abs(df['valence_normalized'] - df['predicted_valence']))
    mae_arousal = np.mean(np.abs(df['arousal_normalized'] - df['predicted_arousal']))
    
    # Mean Squared Error (MSE)
    mse_valence = np.mean((df['valence_normalized'] - df['predicted_valence']) ** 2)
    mse_arousal = np.mean((df['arousal_normalized'] - df['predicted_arousal']) ** 2)
    
    # Root Mean Squared Error (RMSE)
    rmse_valence = np.sqrt(mse_valence)
    rmse_arousal = np.sqrt(mse_arousal)
    
    # Pearson Correlation
    corr_valence = np.corrcoef(df['valence_normalized'], df['predicted_valence'])[0, 1]
    corr_arousal = np.corrcoef(df['arousal_normalized'], df['predicted_arousal'])[0, 1]
    
    # R² Score
    ss_res_valence = np.sum((df['valence_normalized'] - df['predicted_valence']) ** 2)
    ss_tot_valence = np.sum((df['valence_normalized'] - df['valence_normalized'].mean()) ** 2)
    r2_valence = 1 - (ss_res_valence / ss_tot_valence) if ss_tot_valence != 0 else 0
    
    ss_res_arousal = np.sum((df['arousal_normalized'] - df['predicted_arousal']) ** 2)
    ss_tot_arousal = np.sum((df['arousal_normalized'] - df['arousal_normalized'].mean()) ** 2)
    r2_arousal = 1 - (ss_res_arousal / ss_tot_arousal) if ss_tot_arousal != 0 else 0
    
    return {
        'mae_valence': mae_valence,
        'mae_arousal': mae_arousal,
        'mse_valence': mse_valence,
        'mse_arousal': mse_arousal,
        'rmse_valence': rmse_valence,
        'rmse_arousal': rmse_arousal,
        'corr_valence': corr_valence,
        'corr_arousal': corr_arousal,
        'r2_valence': r2_valence,
        'r2_arousal': r2_arousal,
    }


def print_metrics(metrics: Dict[str, float]):
    """Print evaluation metrics in a formatted table."""
    print("\n" + "=" * 60)
    print("📊 EVALUATION METRICS")
    print("=" * 60)
    
    print("\n📈 Valence Metrics:")
    print(f"  MAE:         {metrics['mae_valence']:.4f}")
    print(f"  RMSE:        {metrics['rmse_valence']:.4f}")
    print(f"  Correlation: {metrics['corr_valence']:.4f}")
    print(f"  R² Score:    {metrics['r2_valence']:.4f}")
    
    print("\n📈 Arousal Metrics:")
    print(f"  MAE:         {metrics['mae_arousal']:.4f}")
    print(f"  RMSE:        {metrics['rmse_arousal']:.4f}")
    print(f"  Correlation: {metrics['corr_arousal']:.4f}")
    print(f"  R² Score:    {metrics['r2_arousal']:.4f}")
    
    print("\n📊 Average Metrics:")
    avg_mae = (metrics['mae_valence'] + metrics['mae_arousal']) / 2
    avg_rmse = (metrics['rmse_valence'] + metrics['rmse_arousal']) / 2
    avg_corr = (metrics['corr_valence'] + metrics['corr_arousal']) / 2
    avg_r2 = (metrics['r2_valence'] + metrics['r2_arousal']) / 2
    
    print(f"  MAE:         {avg_mae:.4f}")
    print(f"  RMSE:        {avg_rmse:.4f}")
    print(f"  Correlation: {avg_corr:.4f}")
    print(f"  R² Score:    {avg_r2:.4f}")
    print("=" * 60)


def create_visualizations(df: pd.DataFrame, metrics: Dict[str, float], 
                         output_dir: str):
    """
    Create visualization plots comparing predictions with ground truth.
    
    Args:
        df: DataFrame with predictions and ground truth
        metrics: Dictionary of calculated metrics
        output_dir: Directory to save plots
    """
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (15, 10)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Scatter plots with regression lines
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Valence scatter plot
    axes[0, 0].scatter(df['valence_normalized'], df['predicted_valence'], 
                       alpha=0.6, s=100, c='blue', edgecolors='black')
    axes[0, 0].plot([-1, 1], [-1, 1], 'r--', lw=2, label='Perfect Prediction')
    
    # Add regression line
    z = np.polyfit(df['valence_normalized'], df['predicted_valence'], 1)
    p = np.poly1d(z)
    axes[0, 0].plot(df['valence_normalized'].sort_values(), 
                    p(df['valence_normalized'].sort_values()), 
                    "g-", lw=2, label='Regression Line')
    
    axes[0, 0].set_xlabel('Ground Truth Valence', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Predicted Valence', fontsize=12, fontweight='bold')
    axes[0, 0].set_title(f'Valence: Predictions vs Ground Truth\n' + 
                         f'R²={metrics["r2_valence"]:.3f}, Corr={metrics["corr_valence"]:.3f}, MAE={metrics["mae_valence"]:.3f}',
                         fontsize=13, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xlim(-1.1, 1.1)
    axes[0, 0].set_ylim(-1.1, 1.1)
    
    # Arousal scatter plot
    axes[0, 1].scatter(df['arousal_normalized'], df['predicted_arousal'], 
                       alpha=0.6, s=100, c='red', edgecolors='black')
    axes[0, 1].plot([-1, 1], [-1, 1], 'r--', lw=2, label='Perfect Prediction')
    
    # Add regression line
    z = np.polyfit(df['arousal_normalized'], df['predicted_arousal'], 1)
    p = np.poly1d(z)
    axes[0, 1].plot(df['arousal_normalized'].sort_values(), 
                    p(df['arousal_normalized'].sort_values()), 
                    "g-", lw=2, label='Regression Line')
    
    axes[0, 1].set_xlabel('Ground Truth Arousal', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Predicted Arousal', fontsize=12, fontweight='bold')
    axes[0, 1].set_title(f'Arousal: Predictions vs Ground Truth\n' + 
                         f'R²={metrics["r2_arousal"]:.3f}, Corr={metrics["corr_arousal"]:.3f}, MAE={metrics["mae_arousal"]:.3f}',
                         fontsize=13, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlim(-1.1, 1.1)
    axes[0, 1].set_ylim(-1.1, 1.1)
    
    # Residual plots
    valence_residuals = df['predicted_valence'] - df['valence_normalized']
    arousal_residuals = df['predicted_arousal'] - df['arousal_normalized']
    
    axes[1, 0].scatter(df['valence_normalized'], valence_residuals, 
                       alpha=0.6, s=100, c='blue', edgecolors='black')
    axes[1, 0].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1, 0].set_xlabel('Ground Truth Valence', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Residual (Predicted - Truth)', fontsize=12, fontweight='bold')
    axes[1, 0].set_title(f'Valence Residual Plot\nMean Residual: {valence_residuals.mean():.3f}',
                         fontsize=13, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].scatter(df['arousal_normalized'], arousal_residuals, 
                       alpha=0.6, s=100, c='red', edgecolors='black')
    axes[1, 1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1, 1].set_xlabel('Ground Truth Arousal', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Residual (Predicted - Truth)', fontsize=12, fontweight='bold')
    axes[1, 1].set_title(f'Arousal Residual Plot\nMean Residual: {arousal_residuals.mean():.3f}',
                         fontsize=13, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    scatter_plot_path = os.path.join(output_dir, 'scatter_plots.png')
    plt.savefig(scatter_plot_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Saved scatter plots: {scatter_plot_path}")
    plt.close()
    
    # 2. Error distribution plots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    axes[0].hist(valence_residuals, bins=15, alpha=0.7, color='blue', edgecolor='black')
    axes[0].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[0].set_xlabel('Residual (Predicted - Truth)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[0].set_title(f'Valence Error Distribution\nStd Dev: {valence_residuals.std():.3f}',
                     fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    axes[1].hist(arousal_residuals, bins=15, alpha=0.7, color='red', edgecolor='black')
    axes[1].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('Residual (Predicted - Truth)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[1].set_title(f'Arousal Error Distribution\nStd Dev: {arousal_residuals.std():.3f}',
                     fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    error_plot_path = os.path.join(output_dir, 'error_distribution.png')
    plt.savefig(error_plot_path, dpi=150, bbox_inches='tight')
    print(f"📊 Saved error distribution: {error_plot_path}")
    plt.close()
    
    # 3. Emotion space plot (2D valence-arousal)
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Ground truth
    axes[0].scatter(df['valence_normalized'], df['arousal_normalized'], 
                   alpha=0.6, s=150, c='green', edgecolors='black', label='Ground Truth')
    axes[0].axhline(y=0, color='gray', linestyle='--', lw=1, alpha=0.5)
    axes[0].axvline(x=0, color='gray', linestyle='--', lw=1, alpha=0.5)
    axes[0].set_xlabel('Valence', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Arousal', fontsize=12, fontweight='bold')
    axes[0].set_title('Ground Truth Emotion Space', fontsize=14, fontweight='bold')
    axes[0].set_xlim(-1.1, 1.1)
    axes[0].set_ylim(-1.1, 1.1)
    axes[0].grid(True, alpha=0.3)
    
    # Add quadrant labels
    axes[0].text(0.7, 0.7, 'Happy\nExcited', ha='center', va='center', 
                fontsize=10, alpha=0.6, fontweight='bold')
    axes[0].text(-0.7, 0.7, 'Angry\nTense', ha='center', va='center', 
                fontsize=10, alpha=0.6, fontweight='bold')
    axes[0].text(-0.7, -0.7, 'Sad\nDepressed', ha='center', va='center', 
                fontsize=10, alpha=0.6, fontweight='bold')
    axes[0].text(0.7, -0.7, 'Calm\nPeaceful', ha='center', va='center', 
                fontsize=10, alpha=0.6, fontweight='bold')
    
    # Predictions
    axes[1].scatter(df['predicted_valence'], df['predicted_arousal'], 
                   alpha=0.6, s=150, c='purple', edgecolors='black', label='Predictions')
    axes[1].axhline(y=0, color='gray', linestyle='--', lw=1, alpha=0.5)
    axes[1].axvline(x=0, color='gray', linestyle='--', lw=1, alpha=0.5)
    axes[1].set_xlabel('Valence', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Arousal', fontsize=12, fontweight='bold')
    axes[1].set_title('Predicted Emotion Space', fontsize=14, fontweight='bold')
    axes[1].set_xlim(-1.1, 1.1)
    axes[1].set_ylim(-1.1, 1.1)
    axes[1].grid(True, alpha=0.3)
    
    # Add quadrant labels
    axes[1].text(0.7, 0.7, 'Happy\nExcited', ha='center', va='center', 
                fontsize=10, alpha=0.6, fontweight='bold')
    axes[1].text(-0.7, 0.7, 'Angry\nTense', ha='center', va='center', 
                fontsize=10, alpha=0.6, fontweight='bold')
    axes[1].text(-0.7, -0.7, 'Sad\nDepressed', ha='center', va='center', 
                fontsize=10, alpha=0.6, fontweight='bold')
    axes[1].text(0.7, -0.7, 'Calm\nPeaceful', ha='center', va='center', 
                fontsize=10, alpha=0.6, fontweight='bold')
    
    plt.tight_layout()
    emotion_space_path = os.path.join(output_dir, 'emotion_space.png')
    plt.savefig(emotion_space_path, dpi=150, bbox_inches='tight')
    print(f"📊 Saved emotion space plot: {emotion_space_path}")
    plt.close()
    
    # 4. Comparison plot with arrows
    fig, ax = plt.subplots(figsize=(12, 12))
    
    for idx, row in df.iterrows():
        # Plot ground truth
        ax.scatter(row['valence_normalized'], row['arousal_normalized'], 
                  s=150, c='green', alpha=0.6, edgecolors='black', zorder=3)
        
        # Plot prediction
        ax.scatter(row['predicted_valence'], row['predicted_arousal'], 
                  s=150, c='red', alpha=0.6, edgecolors='black', zorder=3)
        
        # Draw arrow from ground truth to prediction
        ax.arrow(row['valence_normalized'], row['arousal_normalized'],
                row['predicted_valence'] - row['valence_normalized'],
                row['predicted_arousal'] - row['arousal_normalized'],
                head_width=0.03, head_length=0.02, fc='blue', ec='blue', 
                alpha=0.3, zorder=2)
    
    # Add legend (create dummy handles for arrow)
    ax.scatter([], [], s=150, c='green', alpha=0.6, edgecolors='black', label='Ground Truth')
    ax.scatter([], [], s=150, c='red', alpha=0.6, edgecolors='black', label='Predictions')
    from matplotlib.patches import FancyArrow
    arrow_legend = FancyArrow(0, 0, 0.1, 0.1, width=0.02, fc='blue', ec='blue', alpha=0.5)
    ax.add_patch(arrow_legend)
    arrow_legend.set_visible(False)  # Hide the dummy arrow
    # Add manual legend entry for arrow
    from matplotlib.lines import Line2D
    arrow_line = Line2D([0], [0], color='blue', linewidth=2, alpha=0.5, label='Error Vector')
    
    ax.axhline(y=0, color='gray', linestyle='--', lw=1, alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', lw=1, alpha=0.5)
    ax.set_xlabel('Valence', fontsize=14, fontweight='bold')
    ax.set_ylabel('Arousal', fontsize=14, fontweight='bold')
    ax.set_title('Ground Truth vs Predictions in Emotion Space\n(Arrows show prediction error)', 
                fontsize=15, fontweight='bold')
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.grid(True, alpha=0.3)
    
    # Create custom legend handles
    from matplotlib.patches import Patch
    legend_handles = [
        plt.scatter([], [], s=150, c='green', alpha=0.6, edgecolors='black'),
        plt.scatter([], [], s=150, c='red', alpha=0.6, edgecolors='black'),
        Line2D([0], [0], color='blue', linewidth=2, alpha=0.5)
    ]
    ax.legend(legend_handles, ['Ground Truth', 'Predictions', 'Error Vector'], 
             loc='upper left', fontsize=11)
    
    # Add quadrant labels
    ax.text(0.7, 0.7, 'Happy\nExcited', ha='center', va='center', 
           fontsize=11, alpha=0.5, fontweight='bold')
    ax.text(-0.7, 0.7, 'Angry\nTense', ha='center', va='center', 
           fontsize=11, alpha=0.5, fontweight='bold')
    ax.text(-0.7, -0.7, 'Sad\nDepressed', ha='center', va='center', 
           fontsize=11, alpha=0.5, fontweight='bold')
    ax.text(0.7, -0.7, 'Calm\nPeaceful', ha='center', va='center', 
           fontsize=11, alpha=0.5, fontweight='bold')
    
    plt.tight_layout()
    comparison_path = os.path.join(output_dir, 'prediction_comparison.png')
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    print(f"📊 Saved comparison plot: {comparison_path}")
    plt.close()
    
    print(f"\n✅ All visualizations saved to: {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate emotion prediction model against DEAM annotations'
    )
    parser.add_argument(
        '--annotation_file',
        type=str,
        default='../dataset/DEAM/annotations/annotations averaged per song/song_level/static_annotations_averaged_songs_1_2000.csv',
        help='Path to DEAM annotation file'
    )
    parser.add_argument(
        '--audio_dir',
        type=str,
        default='../dataset/DEAM/MEMD_audio',
        help='Directory containing audio files'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default='../selected/final_best_vit',
        help='Path to model checkpoint directory'
    )
    parser.add_argument(
        '--n_samples',
        type=int,
        default=20,
        help='Number of songs to evaluate (default: 20)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./evaluation_results',
        help='Directory to save results and plots'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device to run inference on'
    )
    parser.add_argument(
        '--random_seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🎵 MODEL EVALUATION AGAINST DEAM ANNOTATIONS")
    print("=" * 60)
    
    # Set random seed
    np.random.seed(args.random_seed)
    
    # Load annotations
    annotations = load_deam_annotations(args.annotation_file)
    
    # Normalize DEAM scores to [-1, 1]
    annotations = normalize_deam_scores(annotations)
    
    # Randomly sample songs
    if len(annotations) > args.n_samples:
        sampled_annotations = annotations.sample(n=args.n_samples, random_state=args.random_seed)
        print(f"\n🎲 Randomly selected {args.n_samples} songs for evaluation")
    else:
        sampled_annotations = annotations
        print(f"\n📝 Using all {len(annotations)} annotated songs")
    
    song_ids = sampled_annotations['song_id'].tolist()
    print(f"   Song IDs: {song_ids}")
    
    # Predict emotions
    predictions = predict_emotions(
        audio_dir=args.audio_dir,
        song_ids=song_ids,
        model_path=args.model_path,
        device=args.device
    )
    
    if len(predictions) == 0:
        print("❌ No predictions were made. Check if audio files exist.")
        return
    
    # Merge predictions with annotations
    results = pd.merge(
        sampled_annotations,
        predictions,
        on='song_id',
        how='inner'
    )
    
    print(f"\n✅ Successfully evaluated {len(results)} songs")
    
    # Save detailed results
    os.makedirs(args.output_dir, exist_ok=True)
    results_file = os.path.join(args.output_dir, 'evaluation_results.csv')
    results.to_csv(results_file, index=False)
    print(f"\n💾 Saved detailed results: {results_file}")
    
    # Calculate metrics
    metrics = calculate_metrics(results)
    print_metrics(metrics)
    
    # Save metrics to file
    metrics_file = os.path.join(args.output_dir, 'metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write("EVALUATION METRICS\n")
        f.write("=" * 60 + "\n\n")
        f.write("Valence Metrics:\n")
        f.write(f"  MAE:         {metrics['mae_valence']:.4f}\n")
        f.write(f"  RMSE:        {metrics['rmse_valence']:.4f}\n")
        f.write(f"  Correlation: {metrics['corr_valence']:.4f}\n")
        f.write(f"  R² Score:    {metrics['r2_valence']:.4f}\n\n")
        f.write("Arousal Metrics:\n")
        f.write(f"  MAE:         {metrics['mae_arousal']:.4f}\n")
        f.write(f"  RMSE:        {metrics['rmse_arousal']:.4f}\n")
        f.write(f"  Correlation: {metrics['corr_arousal']:.4f}\n")
        f.write(f"  R² Score:    {metrics['r2_arousal']:.4f}\n\n")
        avg_mae = (metrics['mae_valence'] + metrics['mae_arousal']) / 2
        avg_rmse = (metrics['rmse_valence'] + metrics['rmse_arousal']) / 2
        avg_corr = (metrics['corr_valence'] + metrics['corr_arousal']) / 2
        avg_r2 = (metrics['r2_valence'] + metrics['r2_arousal']) / 2
        f.write("Average Metrics:\n")
        f.write(f"  MAE:         {avg_mae:.4f}\n")
        f.write(f"  RMSE:        {avg_rmse:.4f}\n")
        f.write(f"  Correlation: {avg_corr:.4f}\n")
        f.write(f"  R² Score:    {avg_r2:.4f}\n")
    
    print(f"💾 Saved metrics: {metrics_file}")
    
    # Create visualizations
    create_visualizations(results, metrics, args.output_dir)
    
    print("\n" + "=" * 60)
    print("✅ EVALUATION COMPLETE!")
    print("=" * 60)
    print(f"\n📂 All results saved to: {args.output_dir}/")
    print("\nGenerated files:")
    print(f"  - evaluation_results.csv (detailed predictions)")
    print(f"  - metrics.txt (performance metrics)")
    print(f"  - scatter_plots.png (prediction vs truth)")
    print(f"  - error_distribution.png (error histograms)")
    print(f"  - emotion_space.png (2D emotion space)")
    print(f"  - prediction_comparison.png (arrows showing errors)")
    print()


if __name__ == '__main__':
    main()
