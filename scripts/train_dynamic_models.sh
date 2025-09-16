#!/bin/bash

# Dynamic Models Training Script
# This script trains models for dynamic emotion prediction using different algorithms
# Models trained: linear regression (with Ridge), SVR, and XGBoost
# 
# The dynamic models predict time-varying emotions throughout a song,
# capturing how emotional content changes over time.

# Navigate to the project root directory
cd "$(dirname "$0")/.."

# Create required directories
mkdir -p output/models output/results logs

# Set paths
ANNOTATIONS="annotations/annotations per each rater/dynamic_annotations.csv"
FEATURES="selected"
OUTPUT="output_dynamic"

echo "==========================================="
echo "Starting Dynamic Emotion Models Training"
echo "==========================================="
echo "Annotations: $ANNOTATIONS"
echo "Features: $FEATURES"
echo "Output: $OUTPUT"

# Train all model types on dynamic data with the --dynamic flag
python src/main.py --annotations "$ANNOTATIONS" --features "$FEATURES" \
    --models linear svr xgboost --dynamic --output "$OUTPUT"

# Check if training was successful
if [ $? -eq 0 ]; then
    echo "Dynamic emotion models training completed successfully!"
else
    echo "Error: Dynamic emotion models training failed!"
    exit 1
fi

echo "Models and evaluation results saved to $OUTPUT directory"
echo "==========================================="
