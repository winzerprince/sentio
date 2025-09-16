"""
Data Loader Module for Emotion Prediction from Audio Features

This module provides functionality to:
1. Load annotation data for emotions (both static and dynamic)
2. Load feature data from CSV files in the selected directory
3. Merge annotation and feature data for model training

Key components:
- DataLoader class: Handles loading and merging of annotation and feature data
- Feature processing: Aggregates features for static emotions or aligns by timestamps for dynamic
- Data preparation: Creates feature matrices and target vectors for model training

Notes:
- Static emotions: Single value per song (e.g., overall valence)
- Dynamic emotions: Time-varying values throughout the song (requires timestamp alignment)
- Features are loaded from individual CSV files named with song IDs
"""

import os
import pandas as pd
import numpy as np
from glob import glob
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, annotations_path, features_dir="selected"):
        """
        Initialize the data loader.
        
        Args:
            annotations_path: Path to the annotations CSV file
            features_dir: Directory containing feature CSV files
        """
        self.annotations_path = annotations_path
        self.features_dir = features_dir
        self.annotations = None
        self.features = {}
        
        logger.info(f"DataLoader initialized with annotations: {annotations_path}, features: {features_dir}")
        
    def _scale_annotations(self, annotations):
        """
        Scales valence and arousal from [1, 9] to [-1, 1].
        """
        logger.info("Scaling annotations from [1, 9] to [-1, 1]")
        for col in ['valence', 'arousal']:
            if col in annotations.columns:
                # Formula: new = (val - 5) / 4
                annotations[col] = (annotations[col] - 5.0) / 4.0
        return annotations

    def load_annotations(self, use_dynamic=False):
        """
        Load annotations from file.
        
        Args:
            use_dynamic: If True, use dynamic emotional annotations, otherwise use static
            
        Returns:
            DataFrame of annotations
        """
        try:
            logger.info(f"Loading annotations from: {self.annotations_path}")
            annotations_df = pd.read_csv(self.annotations_path)

            # Scale the annotations from [1, 9] to [-1, 1]
            self.annotations = self._scale_annotations(annotations_df)
            
            # Log information about the loaded annotations
            logger.info(f"Loaded annotations shape: {self.annotations.shape}")
            logger.info(f"Annotations columns: {self.annotations.columns.tolist()}")
            
            # Select appropriate columns based on whether we want static or dynamic data
            if use_dynamic:
                # Assuming dynamic columns have a specific pattern or suffix
                # This needs to be adjusted based on your actual column naming convention
                emotion_cols = [col for col in self.annotations.columns 
                               if col.startswith('dynamic_') or 'time_' in col]
                
                if not emotion_cols:
                    # Fallback to columns containing 'time' or numbers, which might indicate dynamic data
                    emotion_cols = [col for col in self.annotations.columns 
                                   if 'time' in col.lower() or any(c.isdigit() for c in col)]
                
                logger.info(f"Selected dynamic emotion columns: {emotion_cols}")
            else:
                # Assuming static columns are direct emotion ratings
                # Adjust based on your actual column names
                possible_emotion_cols = ['valence', 'arousal', 'dominance', 'emotion', 'intensity']
                emotion_cols = [col for col in self.annotations.columns if col.lower() in possible_emotion_cols]
                
                # If no standard emotion columns found, try columns with 'mean' or 'avg' which might indicate aggregated emotions
                if not emotion_cols:
                    emotion_cols = [col for col in self.annotations.columns 
                                   if 'mean' in col.lower() or 'avg' in col.lower()]
                
                logger.info(f"Selected static emotion columns: {emotion_cols}")
            
            # Ensure we have both song_id and at least one emotion column
            required_cols = ['song_id'] + emotion_cols
            
            # Check if the song_id column exists or try alternatives
            if 'song_id' not in self.annotations.columns:
                # Look for alternatives like 'id', 'track', 'song', etc.
                id_alternatives = ['id', 'track', 'track_id', 'song', 'filename']
                for alt in id_alternatives:
                    if alt in self.annotations.columns:
                        logger.info(f"Using '{alt}' as song_id column")
                        self.annotations['song_id'] = self.annotations[alt]
                        break
                
                if 'song_id' not in self.annotations.columns:
                    # If still no song_id, create one from the index as a last resort
                    logger.warning("No song_id column found, creating from index")
                    self.annotations['song_id'] = self.annotations.index.astype(str)
            
            # Check if all required columns are present
            missing_cols = [col for col in required_cols if col not in self.annotations.columns]
            if missing_cols:
                logger.error(f"Missing required columns: {missing_cols}")
                raise ValueError(f"Annotations file missing required columns: {missing_cols}")
            
            return self.annotations[required_cols]
            
        except Exception as e:
            logger.error(f"Error loading annotations: {e}")
            raise
    
    def load_features(self, selected_ids=None):
        """
        Load features from CSV files in the selected folder.
        
        Args:
            selected_ids: Optional list of song IDs to load. If None, load all.
            
        Returns:
            Dictionary mapping song IDs to feature DataFrames
        """
        try:
            # Get all feature files matching the pattern
            feature_files = glob(os.path.join(self.features_dir, "*_selected.csv"))
            logger.info(f"Found {len(feature_files)} feature files in {self.features_dir}")
            
            # Check if we found any files
            if not feature_files:
                alternative_paths = [
                    os.path.join(self.features_dir, "*.csv"),
                    os.path.join("data/processed/features", "*_selected.csv"),
                    os.path.join("data/processed/features", "*.csv"),
                    os.path.join("data", "selected", "*_selected.csv"),
                    os.path.join("selected", "*_selected.csv")
                ]
                
                for alt_path in alternative_paths:
                    alt_files = glob(alt_path)
                    if alt_files:
                        feature_files = alt_files
                        logger.info(f"Using alternative path: {alt_path}, found {len(feature_files)} files")
                        break
                        
                if not feature_files:
                    logger.error(f"No feature files found in {self.features_dir} or alternative paths")
                    raise FileNotFoundError(f"No feature files found in {self.features_dir}")
            
            # Load each feature file
            for file_path in feature_files:
                try:
                    logger.debug(f"Loading features from: {file_path}")
                    df = pd.read_csv(file_path)
                    
                    # Extract song ID from filename
                    basename = os.path.basename(file_path)
                    song_id = basename.split('_')[0]
                    
                    # Skip if not in selected_ids (if specified)
                    if selected_ids is not None and song_id not in selected_ids:
                        continue
                    
                    # Store the features
                    self.features[song_id] = df
                    
                except Exception as e:
                    logger.warning(f"Error loading {file_path}: {e}")
            
            logger.info(f"Successfully loaded features for {len(self.features)} songs")
            
            # Check if we loaded any features
            if not self.features:
                logger.error("No features were successfully loaded")
                raise ValueError("Failed to load any feature files")
                
            # Sample the first feature file to show its structure
            if self.features:
                sample_id = list(self.features.keys())[0]
                sample_df = self.features[sample_id]
                logger.info(f"Sample feature file ({sample_id}) shape: {sample_df.shape}")
                logger.info(f"Sample feature columns: {sample_df.columns.tolist()[:5]}...")
                
            return self.features
            
        except Exception as e:
            logger.error(f"Error loading features: {e}")
            raise
    
    def _process_static_features(self, song_features):
        """
        Process features for static emotion prediction by aggregating them.
        
        Args:
            song_features: DataFrame containing features for a song
            
        Returns:
            Numpy array of aggregated features
        """
        # For static emotions, we need to aggregate features
        # This could be mean, median, std, min, max, etc.
        agg_features = []
        
        # Add mean of each numerical column
        means = song_features.mean(numeric_only=True)
        agg_features.extend(means.values)
        
        # Add standard deviation of each numerical column
        stds = song_features.std(numeric_only=True)
        agg_features.extend(stds.values)
        
        # Add min/max of each numerical column for range
        mins = song_features.min(numeric_only=True)
        maxs = song_features.max(numeric_only=True)
        agg_features.extend(mins.values)
        agg_features.extend(maxs.values)
        
        # Return aggregated features as a flat array
        return np.array(agg_features)
    
    def _process_dynamic_features(self, song_features, timestamps=None):
        """
        Process features for dynamic emotion prediction, aligning by timestamps if provided.
        
        Args:
            song_features: DataFrame containing features for a song
            timestamps: Optional list of timestamps to align with
            
        Returns:
            Numpy array of processed features
        """
        # For dynamic emotions, we need to align features with emotion timestamps
        # If no timestamps provided, we return the raw features
        if timestamps is None:
            return song_features.values
            
        # If timestamps are provided, we need to align features with them
        # This requires a 'time' or similar column in the features
        time_cols = [col for col in song_features.columns if 'time' in col.lower()]
        
        if not time_cols:
            logger.warning("No time column found in features for dynamic alignment")
            # If no time column, just return evenly spaced samples
            rows = np.linspace(0, len(song_features)-1, len(timestamps), dtype=int)
            return song_features.iloc[rows].values
            
        # Use the first time column for alignment
        time_col = time_cols[0]
        feature_times = song_features[time_col].values
        
        # Find nearest feature rows for each timestamp
        aligned_indices = []
        for ts in timestamps:
            idx = np.abs(feature_times - ts).argmin()
            aligned_indices.append(idx)
            
        # Return aligned features
        return song_features.iloc[aligned_indices].values
    
    def merge_data(self, use_dynamic=False):
        """
        Merge annotations and features.
        
        Args:
            use_dynamic: Whether to use dynamic annotations
            
        Returns:
            X: Feature matrix
            y: Target values
            song_ids: List of song IDs corresponding to each sample
        """
        try:
            # Load annotations if not already loaded
            if self.annotations is None:
                self.load_annotations(use_dynamic)
                
            # Get unique song IDs from annotations
            song_ids_list = self.annotations['song_id'].unique()
            logger.info(f"Found {len(song_ids_list)} unique song IDs in annotations")
            
            # Load features if not already loaded
            if not self.features:
                self.load_features([str(sid) for sid in song_ids_list])
            
            # Create feature matrix and target vector
            X = []
            y = []
            included_ids = []
            
            # Identify emotion columns in annotations
            if use_dynamic:
                # For dynamic emotions, look for columns with time information
                emotion_cols = [col for col in self.annotations.columns 
                               if col not in ['song_id'] and ('dynamic' in col or 'time' in col)]
            else:
                # For static emotions, look for standard emotion dimension names
                possible_emotion_cols = ['valence', 'arousal', 'dominance', 'emotion', 'intensity']
                emotion_cols = [col for col in self.annotations.columns 
                               if col not in ['song_id'] and col.lower() in possible_emotion_cols]
                
                # If standard names not found, try columns with 'mean' or 'avg'
                if not emotion_cols:
                    emotion_cols = [col for col in self.annotations.columns 
                                   if col not in ['song_id'] and ('mean' in col.lower() or 'avg' in col.lower())]
            
            logger.info(f"Using emotion columns: {emotion_cols}")
            
            # Process each song
            for song_id in song_ids_list:
                # Convert song_id to string for dictionary lookup
                str_id = str(song_id)
                
                # Skip if features not available for this song
                if str_id not in self.features:
                    logger.warning(f"No features found for song ID: {str_id}")
                    continue
                
                # Get features for this song
                song_features = self.features[str_id]
                
                # Get annotation rows for this song
                song_annotations = self.annotations[self.annotations['song_id'] == song_id]
                
                if use_dynamic:
                    # For dynamic emotions, we need to align by timestamps
                    # This implementation depends on your data structure
                    # Assuming annotations have timestamps and emotions for different points in time
                    
                    # Check if we have timestamp columns
                    time_cols = [col for col in song_annotations.columns if 'time' in col.lower()]
                    
                    if time_cols:
                        # Use timestamps for alignment
                        timestamps = song_annotations[time_cols[0]].values
                        
                        # Process features for dynamic prediction (align with timestamps)
                        processed_features = self._process_dynamic_features(song_features, timestamps)
                        
                        # For each timestamp, add a sample
                        for i, row in song_annotations.iterrows():
                            feature_idx = i - song_annotations.index[0]  # Relative index in the song annotations
                            if feature_idx < len(processed_features):
                                # Get features for this timestamp
                                features = processed_features[feature_idx]
                                
                                # Get emotion values
                                emotions = [row[col] for col in emotion_cols if col in row]
                                
                                if len(emotions) == len(emotion_cols):
                                    X.append(features)
                                    y.append(emotions)
                                    included_ids.append(str_id)
                    else:
                        # No timestamps, try to match by using equal-length segments
                        logger.warning(f"No timestamp columns found for dynamic emotions in song {str_id}")
                        
                        # Process all features
                        processed_features = self._process_dynamic_features(song_features)
                        
                        # Get all emotion values
                        all_emotions = song_annotations[emotion_cols].values
                        
                        # Match by length (take the shorter of the two)
                        min_len = min(len(processed_features), len(all_emotions))
                        
                        for i in range(min_len):
                            X.append(processed_features[i])
                            y.append(all_emotions[i])
                            included_ids.append(str_id)
                else:
                    # For static emotions, we aggregate features
                    aggregated_features = self._process_static_features(song_features)
                    
                    # Get emotion values (single row for static emotions)
                    emotions = [song_annotations[col].iloc[0] for col in emotion_cols 
                               if col in song_annotations.columns]
                    
                    # Add sample if we have all emotions
                    if len(emotions) == len(emotion_cols):
                        X.append(aggregated_features)
                        y.append(emotions)
                        included_ids.append(str_id)
            
            # Convert to numpy arrays
            X_array = np.array(X)
            y_array = np.array(y)
            
            logger.info(f"Final dataset: {X_array.shape[0]} samples, {X_array.shape[1]} features")
            logger.info(f"Target shape: {y_array.shape}")
            
            return X_array, y_array, included_ids
            
        except Exception as e:
            logger.error(f"Error merging data: {e}")
            raise
