"""
Music Generation Model Testing Framework
Designed for HP EliteBook 840 G3 constraints
"""

import numpy as np
import pandas as pd
import os
import json
import pickle
import time
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Audio processing
import librosa
import soundfile as sf
from scipy.io import wavfile

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
import pickle

# Music Theory
from music21 import stream, note, chord, duration, tempo, meter, key, pitch
import mido

# Markov Chain
from collections import defaultdict, Counter
import random

# System monitoring
import psutil
import gc

class AudioProcessor:
    """Process audio files for generation models"""
    
    def __init__(self, audio_dir: str, sample_rate: int = 22050):
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        
    def extract_basic_features(self, audio_path: str) -> Dict[str, Any]:
        """Extract basic audio features for generation"""
        try:
            y, sr = librosa.load(audio_path, sr=self.sample_rate, duration=30)  # 30 sec clips
            
            # Extract features
            features = {
                'mfcc': librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13),
                'chroma': librosa.feature.chroma(y=y, sr=sr),
                'spectral_centroid': librosa.feature.spectral_centroid(y=y, sr=sr),
                'spectral_rolloff': librosa.feature.spectral_rolloff(y=y, sr=sr),
                'zero_crossing_rate': librosa.feature.zero_crossing_rate(y),
                'tempo': librosa.beat.tempo(y=y, sr=sr)[0]
            }
            
            return features
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return None
    
    def audio_to_piano_roll(self, audio_path: str, n_notes: int = 88) -> Optional[np.ndarray]:
        """Convert audio to piano roll representation (simplified)"""
        try:
            y, sr = librosa.load(audio_path, sr=self.sample_rate, duration=30)
            
            # Use chroma features as a proxy for pitch content
            chroma = librosa.feature.chroma(y=y, sr=sr, hop_length=512)
            
            # Convert to piano roll-like representation
            # This is a simplified approach - real implementation would need pitch detection
            piano_roll = np.zeros((n_notes, chroma.shape[1]))
            
            # Map chroma to piano keys (simplified mapping)
            for t in range(chroma.shape[1]):
                for pitch_class in range(12):
                    if chroma[pitch_class, t] > 0.3:  # Threshold
                        # Map to multiple octaves
                        for octave in range(2, 7):  # C2 to C7
                            note_idx = octave * 12 + pitch_class
                            if note_idx < n_notes:
                                piano_roll[note_idx, t] = chroma[pitch_class, t]
            
            return piano_roll
            
        except Exception as e:
            print(f"Error converting {audio_path} to piano roll: {e}")
            return None

class SimpleMarkovChain:
    """Simple Markov Chain for music generation (lightweight approach)"""
    
    def __init__(self, order: int = 2):
        self.order = order
        self.transitions = defaultdict(Counter)
        self.emotion_models = {}
        
    def train_on_features(self, features_list: List[np.ndarray], 
                         emotions: List[Tuple[float, float]],
                         emotion_categories: List[str]):
        """Train Markov chain on audio features grouped by emotion"""
        
        # Group by emotion categories
        emotion_groups = defaultdict(list)
        for features, emotion_cat in zip(features_list, emotion_categories):
            emotion_groups[emotion_cat].append(features)
        
        # Train separate model for each emotion
        for emotion_cat, feature_group in emotion_groups.items():
            print(f"Training Markov chain for emotion: {emotion_cat}")
            
            transitions = defaultdict(Counter)
            
            for features in feature_group:
                if features is None:
                    continue
                    
                # Convert features to discrete states (quantization)
                states = self._features_to_states(features)
                
                # Build transition table
                for i in range(len(states) - self.order):
                    current_state = tuple(states[i:i+self.order])
                    next_state = states[i+self.order]
                    transitions[current_state][next_state] += 1
            
            self.emotion_models[emotion_cat] = transitions
            print(f"  - {len(transitions)} unique state transitions")
    
    def _features_to_states(self, features: np.ndarray, n_clusters: int = 50) -> List[int]:
        """Convert continuous features to discrete states"""
        # Flatten feature matrix and quantize
        if features.ndim > 1:
            features_flat = features.mean(axis=0)  # Average over time
        else:
            features_flat = features
            
        # Simple quantization (could use K-means clustering for better results)
        quantized = np.digitize(features_flat, np.linspace(features_flat.min(), 
                                                          features_flat.max(), n_clusters))
        return quantized.tolist()
    
    def generate_sequence(self, emotion_category: str, length: int = 100) -> List[int]:
        """Generate a sequence for given emotion"""
        if emotion_category not in self.emotion_models:
            print(f"No model trained for emotion: {emotion_category}")
            return []
        
        transitions = self.emotion_models[emotion_category]
        
        if not transitions:
            return []
        
        # Start with a random initial state
        current_state = random.choice(list(transitions.keys()))
        sequence = list(current_state)
        
        for _ in range(length - self.order):
            if current_state in transitions:
                # Choose next state based on probabilities
                next_states = transitions[current_state]
                total = sum(next_states.values())
                
                if total == 0:
                    # Fallback to random state
                    current_state = random.choice(list(transitions.keys()))
                    continue
                
                # Weighted random choice
                rand_val = random.random() * total
                cumsum = 0
                next_state = None
                
                for state, count in next_states.items():
                    cumsum += count
                    if rand_val <= cumsum:
                        next_state = state
                        break
                
                if next_state is not None:
                    sequence.append(next_state)
                    current_state = tuple(sequence[-self.order:])
                else:
                    current_state = random.choice(list(transitions.keys()))
            else:
                # Start fresh if we reach a dead end
                current_state = random.choice(list(transitions.keys()))
        
        return sequence
    
    def sequence_to_audio_features(self, sequence: List[int], 
                                  reference_features: np.ndarray) -> np.ndarray:
        """Convert generated sequence back to audio-like features"""
        # This is a simplified reverse mapping
        # In practice, this would need a more sophisticated approach
        
        feature_dim = reference_features.shape[0] if reference_features.ndim > 1 else len(reference_features)
        generated_features = np.zeros((feature_dim, len(sequence)))
        
        for i, state in enumerate(sequence):
            # Map state back to feature space (simplified)
            normalized_state = (state % feature_dim) / feature_dim
            generated_features[:, i] = normalized_state * np.random.randn(feature_dim) * 0.1
        
        return generated_features

class SimpleCVAE:
    """Simplified Conditional Variational Autoencoder (lightweight implementation)"""
    
    def __init__(self, latent_dim: int = 16, feature_dim: int = 128):
        self.latent_dim = latent_dim
        self.feature_dim = feature_dim
        self.encoder = None
        self.decoder = None
        self.is_trained = False
        
    def _build_simple_encoder(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Simple encoder using PCA-like dimensionality reduction"""
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # PCA for encoding
        pca = PCA(n_components=self.latent_dim)
        latent_codes = pca.fit_transform(X_scaled)
        
        # Store for decoding
        self.encoder = {'pca': pca, 'scaler': scaler}
        
        return {
            'latent_codes': latent_codes,
            'explained_variance': pca.explained_variance_ratio_
        }
    
    def _build_simple_decoder(self, latent_codes: np.ndarray, 
                            target_features: np.ndarray) -> Dict[str, Any]:
        """Simple decoder using linear regression"""
        from sklearn.linear_model import Ridge
        
        # Train decoder to reconstruct features from latent codes
        decoder = Ridge(alpha=0.1)
        decoder.fit(latent_codes, target_features)
        
        self.decoder = decoder
        
        return {'decoder': decoder}
    
    def train(self, features_list: List[np.ndarray], 
             emotions: List[Tuple[float, float]]) -> Dict[str, Any]:
        """Train the simplified CVAE"""
        print("Training Simplified CVAE...")
        
        # Prepare data
        valid_features = [f for f in features_list if f is not None]
        valid_emotions = emotions[:len(valid_features)]
        
        if len(valid_features) == 0:
            print("No valid features to train on")
            return {}
        
        # Flatten features for training
        X = []
        y_emotions = []
        
        for features, emotion in zip(valid_features, valid_emotions):
            if features.ndim > 1:
                # Average over time dimension
                feature_vector = features.mean(axis=1) if features.shape[0] < features.shape[1] else features.mean(axis=0)
            else:
                feature_vector = features
            
            X.append(feature_vector)
            y_emotions.append(emotion)
        
        X = np.array(X)
        y_emotions = np.array(y_emotions)
        
        print(f"Training data shape: {X.shape}")
        
        # Build encoder
        encoder_results = self._build_simple_encoder(X)
        latent_codes = encoder_results['latent_codes']
        
        # Augment latent codes with emotion information
        latent_with_emotion = np.concatenate([latent_codes, y_emotions], axis=1)
        
        # Build decoder
        decoder_results = self._build_simple_decoder(latent_with_emotion, X)
        
        self.is_trained = True
        
        return {
            'encoder_results': encoder_results,
            'decoder_results': decoder_results,
            'training_data_shape': X.shape,
            'latent_dim': self.latent_dim
        }
    
    def generate(self, target_valence: float, target_arousal: float, 
                n_samples: int = 1) -> List[np.ndarray]:
        """Generate features for target emotion"""
        if not self.is_trained or self.decoder is None:
            print("Model not trained yet or decoder not available")
            return []
        
        generated = []
        
        for _ in range(n_samples):
            # Sample from latent space
            latent_sample = np.random.randn(self.latent_dim)
            
            # Add emotion conditioning
            emotion_vector = np.array([target_valence, target_arousal])
            conditioned_latent = np.concatenate([latent_sample, emotion_vector]).reshape(1, -1)
            
            # Decode
            try:
                reconstructed = self.decoder.predict(conditioned_latent)[0]
                generated.append(reconstructed)
            except Exception as e:
                print(f"Error during generation: {e}")
                continue
        
        return generated

class GenerationModelTester:
    """Test music generation models"""
    
    def __init__(self, audio_dir: str, annotations_df: pd.DataFrame):
        self.audio_dir = audio_dir
        self.annotations_df = annotations_df
        self.audio_processor = AudioProcessor(audio_dir)
        self.results = {}
        
    def categorize_emotions(self, valence: float, arousal: float) -> str:
        """Categorize emotions into discrete categories"""
        # Convert 0-1 normalized values to categories
        if valence > 0.6 and arousal > 0.6:
            return "happy_energetic"
        elif valence > 0.6 and arousal <= 0.6:
            return "peaceful_content"
        elif valence <= 0.4 and arousal > 0.6:
            return "angry_tense"
        elif valence <= 0.4 and arousal <= 0.4:
            return "sad_calm"
        else:
            return "neutral"
    
    def test_markov_chain(self, max_files: int = 100) -> Dict[str, Any]:
        """Test Markov Chain generation"""
        print("\n=== Testing Markov Chain Generation ===")
        start_time = time.time()
        
        # Load audio files and extract features
        features_list = []
        emotions = []
        emotion_categories = []
        
        audio_files = [f for f in os.listdir(self.audio_dir) if f.endswith('.mp3')][:max_files]
        
        for i, audio_file in enumerate(audio_files):
            if i % 10 == 0:
                print(f"Processing file {i+1}/{len(audio_files)}")
                
            # Get song ID from filename
            try:
                song_id = int(audio_file.replace('.mp3', ''))
            except:
                continue
            
            # Get emotion labels
            if song_id not in self.annotations_df['song_id'].values:
                continue
                
            row = self.annotations_df[self.annotations_df['song_id'] == song_id].iloc[0]
            valence = row['valence_norm']
            arousal = row['arousal_norm']
            emotion_cat = self.categorize_emotions(valence, arousal)
            
            # Extract features
            audio_path = os.path.join(self.audio_dir, audio_file)
            features = self.audio_processor.extract_basic_features(audio_path)
            
            if features is not None:
                # Use MFCC as main feature
                mfcc = features['mfcc']
                features_list.append(mfcc)
                emotions.append((valence, arousal))
                emotion_categories.append(emotion_cat)
        
        print(f"Successfully processed {len(features_list)} files")
        
        if len(features_list) == 0:
            return {"error": "No valid audio files processed"}
        
        # Train Markov Chain
        markov = SimpleMarkovChain(order=2)
        markov.train_on_features(features_list, emotions, emotion_categories)
        
        # Test generation for each emotion category
        generation_results = {}
        for emotion_cat in set(emotion_categories):
            print(f"Generating sequence for {emotion_cat}")
            generated_sequence = markov.generate_sequence(emotion_cat, length=50)
            
            if generated_sequence:
                # Convert back to features (simplified)
                ref_features = features_list[0]  # Use first as reference
                generated_features = markov.sequence_to_audio_features(generated_sequence, ref_features)
                
                generation_results[emotion_cat] = {
                    'sequence_length': len(generated_sequence),
                    'unique_states': len(set(generated_sequence)),
                    'generated_features_shape': generated_features.shape
                }
        
        training_time = time.time() - start_time
        
        results = {
            'model_type': 'markov_chain',
            'training_files': len(features_list),
            'emotion_categories': list(set(emotion_categories)),
            'generation_results': generation_results,
            'training_time_seconds': training_time,
            'memory_usage_gb': psutil.virtual_memory().used / (1024**3)
        }
        
        self.results['markov_chain'] = results
        return results
    
    def test_simple_cvae(self, max_files: int = 50) -> Dict[str, Any]:
        """Test Simplified CVAE generation"""
        print("\n=== Testing Simplified CVAE Generation ===")
        start_time = time.time()
        
        # Check memory constraints
        current_mem = psutil.virtual_memory().used / (1024**3)
        if current_mem > 12:
            print("Skipping CVAE due to memory constraints")
            return {"error": "Insufficient memory"}
        
        # Load and process audio files (smaller subset)
        features_list = []
        emotions = []
        
        audio_files = [f for f in os.listdir(self.audio_dir) if f.endswith('.mp3')][:max_files]
        
        for i, audio_file in enumerate(audio_files):
            if i % 10 == 0:
                print(f"Processing file {i+1}/{len(audio_files)}")
            
            try:
                song_id = int(audio_file.replace('.mp3', ''))
            except:
                continue
            
            if song_id not in self.annotations_df['song_id'].values:
                continue
            
            row = self.annotations_df[self.annotations_df['song_id'] == song_id].iloc[0]
            valence = row['valence_norm']
            arousal = row['arousal_norm']
            
            audio_path = os.path.join(self.audio_dir, audio_file)
            features = self.audio_processor.extract_basic_features(audio_path)
            
            if features is not None:
                # Use MFCC as main feature
                mfcc = features['mfcc']
                features_list.append(mfcc)
                emotions.append((valence, arousal))
        
        print(f"Successfully processed {len(features_list)} files for CVAE")
        
        if len(features_list) < 10:
            return {"error": "Insufficient data for CVAE training"}
        
        # Train CVAE
        cvae = SimpleCVAE(latent_dim=8, feature_dim=128)  # Small dimensions
        training_results = cvae.train(features_list, emotions)
        
        # Test generation
        test_emotions = [
            (0.8, 0.8),  # Happy/Energetic
            (0.2, 0.2),  # Sad/Calm
            (0.5, 0.5),  # Neutral
        ]
        
        generation_results = {}
        for i, (val, ar) in enumerate(test_emotions):
            emotion_name = f"test_emotion_{i+1}"
            generated = cvae.generate(val, ar, n_samples=3)
            
            if generated:
                generation_results[emotion_name] = {
                    'target_valence': val,
                    'target_arousal': ar,
                    'generated_samples': len(generated),
                    'feature_shape': generated[0].shape if generated else None
                }
        
        training_time = time.time() - start_time
        
        results = {
            'model_type': 'simple_cvae',
            'training_files': len(features_list),
            'training_results': training_results,
            'generation_results': generation_results,
            'training_time_seconds': training_time,
            'memory_usage_gb': psutil.virtual_memory().used / (1024**3)
        }
        
        self.results['simple_cvae'] = results
        return results
    
    def run_generation_tests(self) -> Dict[str, Any]:
        """Run all generation model tests"""
        print("Starting music generation model testing...")
        
        # Test Markov Chain (lightweight)
        self.test_markov_chain(max_files=80)
        
        # Test CVAE if memory allows
        current_mem = psutil.virtual_memory().used / (1024**3)
        if current_mem < 10:
            self.test_simple_cvae(max_files=40)
        else:
            print("Skipping CVAE due to memory constraints")
        
        return self.results

class GenerationReportGenerator:
    """Generate reports for generation models"""
    
    def __init__(self, results: Dict[str, Any]):
        self.results = results
    
    def create_generation_comparison(self) -> pd.DataFrame:
        """Create comparison table for generation models"""
        comparison_data = []
        
        for model_name, model_results in self.results.items():
            if 'error' in model_results:
                comparison_data.append({
                    'Model': model_name.upper(),
                    'Status': 'ERROR',
                    'Error': model_results['error'],
                    'Training_Files': 0,
                    'Training_Time_sec': 0,
                    'Memory_Usage_GB': 0,
                    'Generation_Categories': 0
                })
                continue
            
            comparison_data.append({
                'Model': model_name.upper(),
                'Status': 'SUCCESS',
                'Error': 'None',
                'Training_Files': model_results.get('training_files', 0),
                'Training_Time_sec': model_results.get('training_time_seconds', 0),
                'Memory_Usage_GB': model_results.get('memory_usage_gb', 0),
                'Generation_Categories': len(model_results.get('generation_results', {}))
            })
        
        return pd.DataFrame(comparison_data)
    
    def save_generation_report(self, save_path: str):
        """Save detailed generation report"""
        with open(save_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)

def main_generation_test():
    """Main generation testing pipeline"""
    BASE_PATH = "/mnt/sdb8mount/free-explore/class/ai/datasets/sentio/dataset/DEAM"
    AUDIO_DIR = os.path.join(BASE_PATH, "MEMD_audio")
    RESULTS_DIR = "/mnt/sdb8mount/free-explore/class/ai/datasets/sentio/results"
    
    # Load annotations
    ann_path1 = os.path.join(BASE_PATH, "annotations/annotations averaged per song/song_level/static_annotations_averaged_songs_1_2000.csv")
    ann_path2 = os.path.join(BASE_PATH, "annotations/annotations averaged per song/song_level/static_annotations_averaged_songs_2000_2058.csv")
    
    df1 = pd.read_csv(ann_path1, skipinitialspace=True)
    df2 = pd.read_csv(ann_path2, skipinitialspace=True)
    annotations_df = pd.concat([df1, df2], ignore_index=True)
    
    # Normalize emotions
    annotations_df['valence_norm'] = (annotations_df['valence_mean'] - 1) / 8
    annotations_df['arousal_norm'] = (annotations_df['arousal_mean'] - 1) / 8
    
    print("=" * 80)
    print("MUSIC GENERATION - MODEL TESTING")
    print("=" * 80)
    print(f"Audio directory: {AUDIO_DIR}")
    print(f"Available annotations: {len(annotations_df)}")
    print("=" * 80)
    
    # Check if audio directory exists
    if not os.path.exists(AUDIO_DIR):
        print(f"WARNING: Audio directory not found: {AUDIO_DIR}")
        print("Generation testing requires original audio files")
        return
    
    # Run tests
    tester = GenerationModelTester(AUDIO_DIR, annotations_df)
    results = tester.run_generation_tests()
    
    # Generate reports
    report_gen = GenerationReportGenerator(results)
    comparison_df = report_gen.create_generation_comparison()
    
    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    comparison_df.to_csv(os.path.join(RESULTS_DIR, 'generation_model_comparison.csv'), index=False)
    report_gen.save_generation_report(os.path.join(RESULTS_DIR, 'generation_detailed_results.json'))
    
    print("\n" + "=" * 80)
    print("GENERATION TESTING COMPLETED")
    print("=" * 80)
    print(comparison_df.to_string(index=False))
    print("=" * 80)

if __name__ == "__main__":
    main_generation_test()
