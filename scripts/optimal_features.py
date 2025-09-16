from collections import defaultdict
from typing import OrderedDict
import pandas as pd


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

def group_features(columns: list[str]) -> dict[str, list[str]]:
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
                print(e)
        if not placed:
            grouped['Others'].append(l)
    return grouped

def transform_csvs(song_ids: list[int]) -> None:
    """
    before running this function make sure you have a folder of features in your root directory with the individual csv files with ; delimeters. then create a folder called `selected` that will have the newly transformed features csv files with the selected features only.
    :return:
    """
    for song in song_ids:
        df = pd.read_csv(f'features/{song}.csv', sep=';')
        features = df.columns.values[1:].tolist()
        grps = group_features(features)
        selected_features = []
        for grp, members in grps.items():
            if grp == 'Others':
                continue
            selected_features.extend(members)
        df.to_csv(f'selected/_{song}_selected.csv', columns=['frameTime'] + selected_features, index=False)
    print("Transformation complete.")

def get_song_ids() -> list[int]:
    """
    make sure the annotations folder is in your root directory.
    :return:
    """
    from pathlib import Path
    ROOT = Path(__file__).resolve().parent
    static_1_2000 = ROOT / 'annotations' / 'annotations averaged per song' / 'song_level' / 'static_annotations_averaged_songs_1_2000.csv'
    static_2000_2058 = ROOT / 'annotations' / 'annotations averaged per song' / 'song_level' / 'static_annotations_averaged_songs_2000_2058.csv'
    ids_1_2000 = pd.read_csv(static_1_2000)['song_id'].tolist()
    ids_2000_2058 = pd.read_csv(static_2000_2058)['song_id'].tolist()
    return ids_1_2000 + ids_2000_2058

s_ids = get_song_ids()
transform_csvs(s_ids)
print(f"Total songs transformed: {len(s_ids)}")
# groups = group_features(features)

# total_selected = 0
# selected = []
# for group, feats in groups.items():
#     if group == 'Others':
#         continue
#     selected.extend(feats)
#     total_selected += len(feats)
#     examples = ", ".join(feats)
#     print(f"{group}: {len(feats)} features; {examples}")
#
# df.to_csv('_2_optimal_features.csv', columns=['frameTime'] + selected, index=False)
# print(f"Total selected features: {total_selected}")
# print(f"Selected features: {selected}")
