"""
Evaluation Script for Comparing Emotion Prediction Models

This script loads and compares results from different emotion prediction models.
The script:
1. Loads evaluation results from trained models
2. Generates comparative visualizations
3. Provides summary statistics on model performance

Usage:
    python evaluate_results.py --results_dir output/results

Key components:
- load_results: Load evaluation results from CSV files
- generate_comparison_plots: Create visualizations comparing models
- print_summary: Print summary statistics about model performance

Notes:
- Works with results from both static and dynamic emotion prediction models
- Creates various plots to help interpret model performance differences
- Useful for comparing multiple model architectures side by side
"""

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_results(results_dir):
    """
    Load evaluation results from CSV files.
    
    Args:
        results_dir: Directory containing result CSV files
        
    Returns:
        DataFrame with combined results
    """
    try:
        # Look for the combined results file first
        combined_file = os.path.join(results_dir, "all_models_comparison.csv")
        if os.path.exists(combined_file):
            logger.info(f"Loading combined results from {combined_file}")
            return pd.read_csv(combined_file)
            
        # If combined file doesn't exist, load individual result files
        result_files = [f for f in os.listdir(results_dir) if f.endswith('_metrics.csv')]
        
        if not result_files:
            logger.error(f"No result files found in {results_dir}")
            return None
            
        results = []
        for file in result_files:
            filepath = os.path.join(results_dir, file)
            df = pd.read_csv(filepath)
            
            # Extract model name from filename
            model_name = file.replace('_metrics.csv', '')
            df['Model'] = model_name
            
            results.append(df)
            
        return pd.concat(results)
        
    except Exception as e:
        logger.error(f"Error loading results: {e}")
        return None

def generate_comparison_plots(results_df, output_dir):
    """
    Generate plots comparing model performance.
    
    Args:
        results_df: DataFrame with model comparison results
        output_dir: Directory to save plots
    """
    try:
        if results_df is None or len(results_df) == 0:
            logger.error("No results to plot")
            return
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot each metric as a grouped bar chart
        for metric in ['R²', 'RMSE', 'MSE', 'MAE']:
            if metric in results_df.columns:
                plt.figure(figsize=(12, 8))
                
                # Create grouped bar plot
                sns.barplot(x='Dimension', y=metric, hue='Model', data=results_df)
                
                plt.title(f"Model Comparison - {metric}")
                plt.xlabel("Emotion Dimension")
                plt.ylabel(metric)
                
                if metric == 'R²':
                    plt.ylim(0, 1)  # R² typically between 0 and 1
                    
                plt.legend(title="Model Type")
                plt.grid(True, axis='y', linestyle='--', alpha=0.7)
                plt.tight_layout()
                
                # Save the plot
                safe_metric = metric.lower().replace('²', '2')
                plt.savefig(f"{output_dir}/model_comparison_{safe_metric}.png")
                plt.close()
                
        # Generate heatmap of all metrics
        plt.figure(figsize=(14, 10))
        
        # Pivot the data for the heatmap
        if 'R²' in results_df.columns:  # Use R² as a representative metric
            pivot_data = results_df.pivot(index='Model', columns='Dimension', values='R²')
            sns.heatmap(pivot_data, annot=True, cmap='viridis', fmt='.3f')
            plt.title('R² Score by Model and Emotion Dimension')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/r2_heatmap.png")
            plt.close()
        
        logger.info(f"Comparison plots saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Error generating comparison plots: {e}")

def print_summary(results_df):
    """
    Print summary statistics about model performance.
    
    Args:
        results_df: DataFrame with model comparison results
    """
    try:
        if results_df is None or len(results_df) == 0:
            logger.error("No results to summarize")
            return
            
        # Get overall statistics by model
        print("\n=== MODEL PERFORMANCE SUMMARY ===\n")
        
        # For R² (higher is better)
        if 'R²' in results_df.columns:
            r2_by_model = results_df.groupby('Model')['R²'].agg(['mean', 'std', 'min', 'max'])
            print("\nR² Score Summary (higher is better):")
            print(r2_by_model.to_string())
            
            # Get best model by R²
            best_model_r2 = r2_by_model['mean'].idxmax()
            print(f"\nBest model by R² score: {best_model_r2}")
        
        # For RMSE (lower is better)
        if 'RMSE' in results_df.columns:
            rmse_by_model = results_df.groupby('Model')['RMSE'].agg(['mean', 'std', 'min', 'max'])
            print("\nRMSE Summary (lower is better):")
            print(rmse_by_model.to_string())
            
            # Get best model by RMSE
            best_model_rmse = rmse_by_model['mean'].idxmin()
            print(f"\nBest model by RMSE: {best_model_rmse}")
            
        # Performance by dimension
        print("\n=== PERFORMANCE BY EMOTION DIMENSION ===\n")
        
        for dim in results_df['Dimension'].unique():
            dim_data = results_df[results_df['Dimension'] == dim]
            
            print(f"\nDimension: {dim}")
            
            if 'R²' in results_df.columns:
                best_model = dim_data.loc[dim_data['R²'].idxmax()]
                print(f"Best model: {best_model['Model']} (R² = {best_model['R²']:.4f})")
                
            if 'RMSE' in results_df.columns:
                lowest_rmse = dim_data.loc[dim_data['RMSE'].idxmin()]
                print(f"Lowest RMSE: {lowest_rmse['Model']} (RMSE = {lowest_rmse['RMSE']:.4f})")
                
            print("-" * 40)
        
    except Exception as e:
        logger.error(f"Error printing summary: {e}")

def main():
    """Main function for evaluating and comparing models."""
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Evaluate and compare emotion prediction models")
    parser.add_argument("--results_dir", type=str, required=True, 
                        help="Directory containing evaluation results")
    parser.add_argument("--output_dir", type=str, default=None, 
                        help="Directory to save comparison plots (defaults to results_dir)")
    
    args = parser.parse_args()
    
    # Set output directory to results_dir if not specified
    output_dir = args.output_dir if args.output_dir else args.results_dir
    
    logger.info(f"Loading results from {args.results_dir}")
    
    # Load results
    results = load_results(args.results_dir)
    
    # Generate comparison plots
    generate_comparison_plots(results, output_dir)
    
    # Print summary statistics
    print_summary(results)
    
    logger.info("Model evaluation and comparison completed")

if __name__ == "__main__":
    main()
