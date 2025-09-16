from collections import defaultdict
from typing import OrderedDict
import pandas as pd

# before running this file, get a sample csv file in the features directory and change the delimeter to a comma. then place it in this directory
df = pd.read_csv('_2.csv')

features = df.columns.values[1:].tolist()
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
                'mfcc_sma[4]'  in n.lower() or
                'mfcc_sma_de[4]' in n.lower() or
                'mfcc_sma_de[3]' in n.lower() or
                'mfcc_sma_de[2]' in n.lower() or
                'mfcc_sma_de[1]' in n.lower()
        )), # 16
])

def group_features(columns: list[str]) -> dict[str, list[str]]:
    grouped = defaultdict(list)
    for c in columns:
        l = c.lower()
        placed = False
        for group, matcher in optimal_features_groups.items():
            try:
                if matcher(l):
                    grouped[group].append(l)
                    placed = True
                    break
            except Exception as e:
                print(e)
        if not placed:
            grouped['Others'].append(l)
    return grouped

groups = group_features(features)

total_selected = 0
for group, feats in groups.items():
    if group == 'Others':
        continue
    total_selected += len(feats)
    examples = ", ".join(feats)
    print(f"{group}: {len(feats)} features; {examples}")

print(f"Total selected features: {total_selected}")
