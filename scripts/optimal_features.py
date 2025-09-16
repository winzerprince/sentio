from collections import defaultdict
from typing import OrderedDict, List, Dict
import pandas as pd
import os
import sys


# df = pd.read_csv('_2.csv')
# df2 = pd.read_csv('features/2001.csv', sep=';')
# features = df.columns.values[1:].tolist()

optimal_features_groups = OrderedDict([
        ("Energy", lambda n: "rms" in n.lower()), # 4 ,
        ("Spetral Rolloff", lambda n: "rolloff75" in n.lower() or "rolloff90" in n.lower()), # 8
        ("Spectral Dynamics/Flux", lambda n: "spectralflux" in n.lower()), # 4
        ("Spectral Centroid", lambda n: "spectralcentroid" in n.lower()), # 4
        ("Harmonicity", lambda n: "spectralharmonicity" in n.lower()), # 4
        ("MFCC features", lambda n: (
                'mfcc_sma[1]' in n.lower() or
                'mfcc_sma[2]' in n.lower() or
                'mfcc_sma[3]' in n.lower() or
                'mfcc_sma_de[3]' in n.lower() or
                'mfcc_sma_de[2]' in n.lower() or
                'mfcc_sma_de[1]' in n.lower()
        )), # 16
        ("Frequency features", lambda n: n.lower().startswith("f0")) # 2
])

def group_features(columns: List[str]) -> Dict[str, List[str]]:
    """
    Group feature columns based on predefined rules.
    
    Args:
        columns: List of column names to categorize
        
    Returns:
        Dictionary mapping group names to lists of features
    """
    grouped = defaultdict(list)
    for c in columns:
        l = c.lower()
        placed = False
        for group, matcher in optimal_features_groups.items():
            try:
                if matcher(l):
                    grouped[group].append(c)
                    placed = True
                    break
            except Exception as e:
                print(f"Error matching feature '{c}' for group '{group}': {e}")
        if not placed:
            grouped['Others'].append(c)
    return grouped

def transform_csvs(song_ids: List[int], features_dir: str = 'features', output_dir: str = 'selected') -> None:
    """
    Process song feature CSV files by extracting the optimal features and saving to new files.
    
    Args:
        song_ids: List of song IDs to process
        features_dir: Directory containing original feature CSV files
        output_dir: Directory where processed CSV files will be saved
        
    Returns:
        None
    """
    # Check if the features directory exists
    if not os.path.exists(features_dir):
        print(f"Error: Features directory '{features_dir}' not found!")
        print(f"Current working directory: {os.getcwd()}")
        return
    
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        print(f"Creating output directory: {output_dir}")
        try:
            os.makedirs(output_dir)
        except Exception as e:
            print(f"Error creating output directory: {e}")
            return
    
    successful = 0
    failed = 0
    
    for song in song_ids:
        feature_file = f'{features_dir}/{song}.csv'
        output_file = f'{output_dir}/{song}_selected.csv'
        
        try:
            # Check if the feature file exists
            if not os.path.exists(feature_file):
                print(f"Warning: Feature file '{feature_file}' not found. Skipping.")
                failed += 1
                continue
                
            # Read the feature file
            print(f"Processing song {song}...")
            df = pd.read_csv(feature_file, sep=';')
            
            # Extract features
            features = [str(col) for col in df.columns[1:]]
            grps = group_features(features)
            
            # Select features from desired groups
            selected_features = []
            for grp, members in grps.items():
                if grp == 'Others':
                    continue
                selected_features.extend(members)
            
            if not selected_features:
                print(f"Warning: No features selected for song {song}. Skipping.")
                failed += 1
                continue
                
            # Save the selected features
            df.to_csv(output_file, columns=['frameTime'] + selected_features, index=False)
            print(f"Saved selected features for song {song} to {output_file}")
            successful += 1
            
        except Exception as e:
            print(f"Error processing song {song}: {e}")
            failed += 1
    
    print(f"\nSummary: Processed {successful} songs successfully, {failed} songs failed.")

def find_available_song_ids(features_dir: str = 'features') -> List[int]:
    """
    Find all available song IDs in the features directory.
    
    Args:
        features_dir: Directory containing feature CSV files
        
    Returns:
        List of valid song IDs found in the directory
    """
    if not os.path.exists(features_dir):
        print(f"Error: Features directory '{features_dir}' not found!")
        return []
        
    song_ids = []
    for filename in os.listdir(features_dir):
        if filename.endswith('.csv'):
            try:
                # Extract song ID from filename (removing .csv extension)
                song_id = int(filename.split('.')[0])
                song_ids.append(song_id)
            except (ValueError, IndexError):
                # Skip files that don't follow the expected naming pattern
                continue
                
    return song_ids

def main():
    """
    Main function to run the optimal features extraction process.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract optimal features from audio feature CSV files.')
    parser.add_argument('--features-dir', type=str, default='features',
                        help='Directory containing original feature CSV files')
    parser.add_argument('--output-dir', type=str, default='selected',
                        help='Directory where processed CSV files will be saved')
    parser.add_argument('--song-ids', type=int, nargs='+',
                        help='List of song IDs to process (if not provided, all available songs will be processed)')
    
    args = parser.parse_args()
    
    # Validate features directory
    if not os.path.exists(args.features_dir):
        print(f"Error: Features directory '{args.features_dir}' not found!")
        print(f"Current working directory: {os.getcwd()}")
        sys.exit(1)
        
    # Get song IDs
    song_ids = args.song_ids
    if song_ids is None:
        print("No song IDs provided. Finding all available songs...")
        song_ids = find_available_song_ids(args.features_dir)
        if not song_ids:
            print("No valid song files found in the features directory.")
            sys.exit(1)
        print(f"Found {len(song_ids)} songs to process.")
    
    # Process songs
    transform_csvs(song_ids, args.features_dir, args.output_dir)

if __name__ == "__main__":
    main()