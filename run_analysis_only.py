"""
Simplified Analysis-Only Testing Script
Focuses on emotion analysis models without audio generation dependencies
"""

import sys
import os
sys.path.append('src')

from model_testing_framework import main

if __name__ == "__main__":
    print("="*80)
    print("EMOTION ANALYSIS MODEL TESTING (Analysis Only)")
    print("="*80)
    print("System: HP EliteBook 840 G3 (Intel i5-6300U, 16GB RAM)")
    print("Testing: Ridge, SVR, XGBoost, MLP models")
    print("Dataset: DEAM emotion annotations + features")
    print("="*80)
    
    # Run analysis testing only
    main()
