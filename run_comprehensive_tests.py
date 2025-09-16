"""
Comprehensive Model Testing Plan Execution Script
This script runs all analysis and generation model tests in the optimal order
"""

import sys
import os
import subprocess
import time
from datetime import datetime
import psutil

def check_requirements():
    """Check if all required packages are installed"""
    required_packages = [
        'pandas', 'numpy', 'scikit-learn', 'xgboost', 'lightgbm',
        'matplotlib', 'seaborn', 'librosa', 'soundfile', 'music21',
        'mido', 'psutil'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing required packages: {missing_packages}")
        print("Please install them using: pip install " + " ".join(missing_packages))
        return False
    
    return True

def monitor_system():
    """Monitor system resources"""
    memory = psutil.virtual_memory()
    cpu = psutil.cpu_percent(interval=1)
    
    return {
        'memory_used_gb': memory.used / (1024**3),
        'memory_percent': memory.percent,
        'cpu_percent': cpu,
        'available_memory_gb': memory.available / (1024**3)
    }

def run_analysis_tests():
    """Run emotion analysis model tests"""
    print("\n" + "="*80)
    print("PHASE 1: EMOTION ANALYSIS MODEL TESTING")
    print("="*80)
    
    # Import and run analysis tests
    sys.path.append('/mnt/sdb8mount/free-explore/class/ai/datasets/sentio/src')
    
    try:
        from model_testing_framework import main
        main()
        return True
    except Exception as e:
        print(f"Error in analysis testing: {e}")
        return False

def run_generation_tests():
    """Run music generation model tests"""
    print("\n" + "="*80)
    print("PHASE 2: MUSIC GENERATION MODEL TESTING")
    print("="*80)
    
    # Check if we have enough memory for generation tests
    stats = monitor_system()
    if stats['available_memory_gb'] < 4:
        print("WARNING: Low memory available. Generation tests may fail.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return False
    
    try:
        from generation_testing_framework import main_generation_test
        main_generation_test()
        return True
    except Exception as e:
        print(f"Error in generation testing: {e}")
        return False

def create_final_report():
    """Create a comprehensive final report combining both test results"""
    print("\n" + "="*80)
    print("PHASE 3: GENERATING COMPREHENSIVE REPORT")
    print("="*80)
    
    results_dir = "/mnt/sdb8mount/free-explore/class/ai/datasets/sentio/results"
    
    try:
        import pandas as pd
        import json
        from datetime import datetime
        
        # Load analysis results
        analysis_comparison = None
        generation_comparison = None
        
        analysis_file = os.path.join(results_dir, 'model_comparison.csv')
        generation_file = os.path.join(results_dir, 'generation_model_comparison.csv')
        
        if os.path.exists(analysis_file):
            analysis_comparison = pd.read_csv(analysis_file)
            print(f"Loaded analysis results: {len(analysis_comparison)} models tested")
        
        if os.path.exists(generation_file):
            generation_comparison = pd.read_csv(generation_file)
            print(f"Loaded generation results: {len(generation_comparison)} models tested")
        
        # Create comprehensive report
        report = {
            'test_date': datetime.now().isoformat(),
            'system_specs': {
                'model': 'HP EliteBook 840 G3',
                'cpu': 'Intel i5-6300U',
                'ram_gb': 16,
                'gpu': 'Intel HD Graphics 520'
            },
            'dataset_info': {
                'name': 'DEAM (Database for Emotion Analysis using Music)',
                'total_songs': 1802,
                'test_sample_size': 1200,
                'emotion_scale': '1-9 (normalized to 0-1)',
                'features_per_song': '260+'
            }
        }
        
        if analysis_comparison is not None:
            # Find best performing models
            best_valence = analysis_comparison[analysis_comparison['Target'] == 'Valence'].loc[
                analysis_comparison[analysis_comparison['Target'] == 'Valence']['R²'].idxmax()
            ]
            best_arousal = analysis_comparison[analysis_comparison['Target'] == 'Arousal'].loc[
                analysis_comparison[analysis_comparison['Target'] == 'Arousal']['R²'].idxmax()
            ]
            
            report['analysis_results'] = {
                'models_tested': analysis_comparison['Model'].unique().tolist(),
                'best_valence_model': {
                    'model': best_valence['Model'],
                    'r2_score': best_valence['R²'],
                    'rmse': best_valence['RMSE'],
                    'training_time_min': best_valence['Training_Time_min']
                },
                'best_arousal_model': {
                    'model': best_arousal['Model'],
                    'r2_score': best_arousal['R²'],
                    'rmse': best_arousal['RMSE'],
                    'training_time_min': best_arousal['Training_Time_min']
                }
            }
        
        if generation_comparison is not None:
            successful_models = generation_comparison[generation_comparison['Status'] == 'SUCCESS']
            
            report['generation_results'] = {
                'models_tested': generation_comparison['Model'].unique().tolist(),
                'successful_models': successful_models['Model'].tolist() if len(successful_models) > 0 else [],
                'total_training_files': int(successful_models['Training_Files'].sum()) if len(successful_models) > 0 else 0
            }
        
        # Save comprehensive report
        report_file = os.path.join(results_dir, 'comprehensive_test_report.json')
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Comprehensive report saved to: {report_file}")
        
        # Print summary
        print("\n" + "="*80)
        print("FINAL SUMMARY")
        print("="*80)
        
        if 'analysis_results' in report:
            print(f"✓ Analysis Models Tested: {len(report['analysis_results']['models_tested'])}")
            print(f"  Best Valence Model: {report['analysis_results']['best_valence_model']['model']} "
                  f"(R² = {report['analysis_results']['best_valence_model']['r2_score']:.3f})")
            print(f"  Best Arousal Model: {report['analysis_results']['best_arousal_model']['model']} "
                  f"(R² = {report['analysis_results']['best_arousal_model']['r2_score']:.3f})")
        
        if 'generation_results' in report:
            print(f"✓ Generation Models Tested: {len(report['generation_results']['models_tested'])}")
            print(f"  Successful Models: {len(report['generation_results']['successful_models'])}")
        
        print(f"✓ All results saved to: {results_dir}")
        print("="*80)
        
        return True
        
    except Exception as e:
        print(f"Error creating final report: {e}")
        return False

def main():
    """Main execution pipeline"""
    start_time = time.time()
    
    print("="*80)
    print("COMPREHENSIVE AI MUSIC EMOTION ANALYSIS & GENERATION TESTING")
    print("="*80)
    print("System: HP EliteBook 840 G3 (Intel i5-6300U, 16GB RAM)")
    print(f"Start time: {datetime.now()}")
    
    # Check system requirements
    initial_stats = monitor_system()
    print(f"Initial memory usage: {initial_stats['memory_used_gb']:.2f}GB "
          f"({initial_stats['memory_percent']:.1f}%)")
    print(f"Available memory: {initial_stats['available_memory_gb']:.2f}GB")
    
    if initial_stats['available_memory_gb'] < 8:
        print("WARNING: Less than 8GB memory available. Tests may fail.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Check requirements
    if not check_requirements():
        return
    
    # Create results directory
    results_dir = "/mnt/sdb8mount/free-explore/class/ai/datasets/sentio/results"
    os.makedirs(results_dir, exist_ok=True)
    
    success_count = 0
    total_phases = 3
    
    # Phase 1: Analysis models
    if run_analysis_tests():
        success_count += 1
        print("✓ Phase 1 completed successfully")
    else:
        print("✗ Phase 1 failed")
    
    # Phase 2: Generation models
    if run_generation_tests():
        success_count += 1
        print("✓ Phase 2 completed successfully")
    else:
        print("✗ Phase 2 failed")
    
    # Phase 3: Final report
    if create_final_report():
        success_count += 1
        print("✓ Phase 3 completed successfully")
    else:
        print("✗ Phase 3 failed")
    
    # Final statistics
    total_time = time.time() - start_time
    final_stats = monitor_system()
    
    print("\n" + "="*80)
    print("TESTING PIPELINE COMPLETED")
    print("="*80)
    print(f"Phases completed: {success_count}/{total_phases}")
    print(f"Total execution time: {total_time/60:.2f} minutes")
    print(f"Peak memory usage: {final_stats['memory_used_gb']:.2f}GB")
    print(f"Results directory: {results_dir}")
    print("="*80)

if __name__ == "__main__":
    main()
