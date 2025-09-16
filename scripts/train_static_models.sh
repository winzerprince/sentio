#!/bin/bash

# Static Models Training Script
# This script trains models for static emotion prediction using different algorithms
# Models trained: linear regression (with Ridge), SVR, and XGBoost
# 
# The static models predict overall/average emotions for each song, rather than
# time-varying emotional changes throughout the song.

# Navigate to the project root directory
cd "$(dirname "$0")/.."

# Create required directories
mkdir -p output/models output/results logs

# Set paths
ANNOTATIONS="annotations/annotations averaged per song/song_level/static_annotations_averaged_songs_1_2000.csv"
FEATURES="selected"
OUTPUT="output"

echo "========================================"
echo "Starting Static Emotion Models Training"
echo "========================================"
echo "Annotations: $ANNOTATIONS"
echo "Features: $FEATURES"
echo "Output: $OUTPUT"

# Train all model types on static data
python src/main.py --annotations "$ANNOTATIONS" --features "$FEATURES" \
    --models linear svr xgboost --output "$OUTPUT"

# Check if training was successful
if [ $? -eq 0 ]; then
    echo "Static emotion models training completed successfully!"
else
    echo "Error: Static emotion models training failed!"
    exit 1
fi

echo "Models and evaluation results saved to $OUTPUT directory"
echo "========================================"
