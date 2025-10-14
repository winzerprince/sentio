# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import mean_squared_error, mean_absolute_error  # For eval


from google.colab import drive
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import layers, models
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import os

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import librosa
import matplotlib.pyplot as plt
import IPython.display as ipd
import warnings
warnings.filterwarnings('ignore')
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
file_paths = []
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        file_paths.append(os.path.join(dirname, filename))

# Print some paths 
for i in range(2):
    print(file_paths[i])
    print(file_paths[len(file_paths)-1-i])
    print(file_paths[(len(file_paths)//2)-i])

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


from pathlib import Path

# Define the dataset's root path
dataset_path = '/kaggle/input/deam-mediaeval-dataset-emotional-analysis-in-music'

# List all files and subdirectories within the dataset directory
files = os.listdir(dataset_path)
print("Files and subdirectories in the dataset:")
print(files)

print(os.listdir('/kaggle/input'))


# Base dataset path
dataset_path = '/kaggle/input/deam-mediaeval-dataset-emotional-analysis-in-music'

# Define the paths for "DEAM_audio/MEMD_audio" and the static annotations CSV
audio_dir = os.path.join(dataset_path, 'DEAM_audio', 'MEMD_audio')
static_annots = Path('/kaggle/input')
static_csv = static_annots / 'static-annotations-1-2000' / 'static_annotations_averaged_songs_1_2000.csv'

annots_2058 = static_annots / 'static-annots-2058' / 'static_annots_2058.csv'

df1 = pd.read_csv(static_csv) # 1744, 5
df2 = pd.read_csv(annots_2058) # 58, 5
df = pd.concat([df1, df2], axis=0)

df.shape


# Print to verify the paths
# print("Audio Directory Path:", audio_dir)
# print("Static CSV Path:", static_csv)


# Check if the paths exist
# if os.path.exists(audio_dir):
#     print("Audio directory exists.")
# else:
#     print("Audio directory does not exist!")

# if os.path.exists(static_csv):
#     print("Static CSV file exists.")
# else:
#     print("Static CSV file does not exist!")


song_id = 10  # song ID

# Load audio
audio_path = os.path.join(audio_dir, f"{song_id}.mp3")
y, sr = librosa.load(audio_path, sr=44100, mono=True)
print(f"Audio loaded: {len(y)} samples at {sr} Hz")
ipd.Audio(data=y,rate=sr)


# Convert to mel-spectrogram
mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=2048, hop_length=512)
mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
print(f"Mel-spectrogram shape: {mel_spec_db.shape}")

# Load static annotations
# df = pd.read_csv(static_csv)
label = df[df['song_id'] == song_id][['valence_mean', 'arousal_mean']].values[0]
print(f"Valence: {label[0]}, Arousal: {label[1]}")

# Segment into 5-second chunks (for consistency with CNN)
segment_length = 5  # seconds
segment_samples = segment_length * sr
segments = [y[i:i+segment_samples] for i in range(0, len(y), segment_samples)
            if len(y[i:i+segment_samples]) == segment_samples]

mel_specs = []
for segment in segments:
    mel_spec = librosa.feature.melspectrogram(y=segment, sr=sr, n_mels=128, n_fft=2048, hop_length=512)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_specs.append(mel_spec_db)

mel_specs = np.array(mel_specs)
print(f"Number of 5-second segments: {len(mel_specs)}, Shape of each: {mel_specs[0].shape}")


# 1. Visualize full mel-spectrogram
ipd.Audio(data=y,rate=sr) # load
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
librosa.display.specshow(mel_spec_db, sr=sr, hop_length=512, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title(f'Mel-Spectrogram for Song ID {song_id} (Full)')
plt.tight_layout()

# 2. Visualize first segment's mel-spectrogram
plt.subplot(1, 2, 2)
librosa.display.specshow(mel_specs[0], sr=sr, hop_length=512, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title(f'Mel-Spectrogram for Song ID {song_id} (First 5s Segment)')
plt.tight_layout()

# 3. Plot valence-arousal on 2D plane
plt.figure(figsize=(6, 6))
plt.scatter(label[0], label[1], color='red', s=100, label=f'Song {song_id}')
plt.xlabel('Valence Mean (1-9)')
plt.ylabel('Arousal Mean (1-9)')
plt.title(f'Valence-Arousal Plane for Song ID {song_id}')
plt.xlim(1, 9)
plt.ylim(1, 9)
plt.grid(True)
plt.legend()
plt.show()


static_csv = '/kaggle/input/deam-mediaeval-dataset-emotional-analysis-in-music/DEAM_Annotations/annotations/annotations averaged per song/song_level/static_annotations_averaged_songs_1_2000.csv'
df_annotations = df
print(f"Loaded {len(df_annotations)} songs.")

class DEAMSpectrogramDataset(Dataset):
    def __init__(self, df_annotations, audio_dir, segment_length=5, sr=44100, target_time_steps=432, use_aug=False):
        self.df = df_annotations
        self.audio_dir = audio_dir
        self.segment_length = segment_length
        self.sr = sr
        self.segment_samples = segment_length * sr
        self.target_time_steps = target_time_steps  # For padding to fixed width (multiple of patch_size)
        self.use_aug = use_aug  # For SpecAugment
        self.segment_indices = []
        for song_idx in range(len(self.df)):
            song_id = int(self.df.iloc[song_idx]['song_id'])
            audio_path = os.path.join(self.audio_dir, f"{song_id}.mp3")
            try:
                y, _ = librosa.load(audio_path, sr=self.sr, mono=True)
                num_segments = len(y) // self.segment_samples
                for seg_idx in range(num_segments):
                    self.segment_indices.append((song_idx, seg_idx))
            except Exception as e:
                print(f"Error loading song {song_id}: {e}, skipping.")
    
    def __len__(self):
        return len(self.segment_indices)
    
    def __getitem__(self, idx):
        song_idx, seg_idx = self.segment_indices[idx]
        song_id = int(self.df.iloc[song_idx]['song_id'])
        audio_path = os.path.join(self.audio_dir, f"{song_id}.mp3")
        y, _ = librosa.load(audio_path, sr=self.sr, mono=True)
        start = seg_idx * self.segment_samples
        segment = y[start:start + self.segment_samples]
        mel_spec = librosa.feature.melspectrogram(y=segment, sr=self.sr, n_mels=128, n_fft=2048, hop_length=512)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Pad to target_time_steps
        if mel_spec_db.shape[1] < self.target_time_steps:
            pad_width = self.target_time_steps - mel_spec_db.shape[1]
            mel_spec_db = np.pad(mel_spec_db, ((0,0), (0, pad_width)), mode='constant')
        elif mel_spec_db.shape[1] > self.target_time_steps:
            mel_spec_db = mel_spec_db[:, :self.target_time_steps]
        
        mel_spec_db = torch.tensor(mel_spec_db).unsqueeze(0).float()  # (1, 128, 432)
        
        # Optional SpecAugment (random mask time/freq)
        if self.use_aug:
            freq_mask = torch.randint(0, 128-10, (1,)).item()  # Mask start
            time_mask = torch.randint(0, self.target_time_steps-20, (1,)).item()
            mel_spec_db[:, freq_mask:freq_mask+10, :] = 0  # Freq mask
            mel_spec_db[:, :, time_mask:time_mask+20] = 0  # Time mask
        
        label = torch.tensor([self.df.iloc[song_idx]['valence_mean'], 
                              self.df.iloc[song_idx]['arousal_mean']]).float()
        return mel_spec_db, label
    
audio_dir = '/kaggle/input/deam-mediaeval-dataset-emotional-analysis-in-music/DEAM_audio/MEMD_audio'
full_dataset = DEAMSpectrogramDataset(df_annotations, audio_dir, target_time_steps=432, use_aug=False)  # Aug off for stats
print(f"Total segments: {len(full_dataset)}")

# Split first (to compute stats on train only)
train_size = int(0.8 * len(full_dataset))
val_size = int(0.1 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

# Compute global mean/std on train set for normalization
train_specs = [full_dataset[idx][0] for idx in train_dataset.indices]  # Get train specs
train_specs = torch.cat(train_specs)  # (num_train_segments * 1, 128, 432) -> flatten for stats
global_mean = train_specs.mean()
global_std = train_specs.std()
print(f"Global mean/std: {global_mean:.4f}/{global_std:.4f}")

# Simple Test: Load one item
test_spec, test_label = full_dataset[0]
print(f"Test shape: {test_spec.shape}, Label: {test_label}")


# Re-init datasets with aug for train
train_dataset = DEAMSpectrogramDataset(df_annotations.iloc[train_dataset.indices], audio_dir, target_time_steps=432, use_aug=True)
val_dataset = DEAMSpectrogramDataset(df_annotations.iloc[val_dataset.indices], audio_dir, target_time_steps=432, use_aug=False)
test_dataset = DEAMSpectrogramDataset(df_annotations.iloc[test_dataset.indices], audio_dir, target_time_steps=432, use_aug=False)

# Apply normalization in a wrapper or here; for simplicity, add to __getitem__ if needed, but demo in loader test
batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# Simple Test: Batch with norm
batch_specs, _ = next(iter(train_loader))
batch_specs = (batch_specs - global_mean) / global_std  # Apply norm
print(f"Normalized batch shape: {batch_specs.shape}")

class SpectrogramTransformer(nn.Module):
    def __init__(self, input_height=128, input_width=432, patch_size=16, embed_dim=256, num_heads=4, num_layers=4, dropout=0.1):
        super().__init__()
        self.patch_size = patch_size
        num_patches = (input_height // patch_size) * (input_width // patch_size)  # 8 x 27 = 216
        self.patch_embed = nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # Learnable CLS
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + num_patches, embed_dim))  # Pos for CLS + patches
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 128),
            nn.GELU(),  # As in advice
            nn.Linear(128, 2)
        )
    
    def forward(self, x):
        # Normalize if not done (assume done in loader)
        # x = (x - global_mean) / global_std  # Uncomment if needed
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # Prepend CLS
        x = x + self.pos_embed
        x = self.transformer_encoder(x)
        x = x[:, 0]  # Extract CLS token
        return self.head(x)
    
    model = SpectrogramTransformer()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print("Model on", device)

# For transfer learning: Search for pretrained AST (e.g., "audioset ast pytorch"), upload weights to Kaggle, load with model.load_state_dict(torch.load('path', map_location=device), strict=False)

# Simple Test
test_input = torch.randn(2, 1, 128, 432).to(device)
test_output = model(test_input)
print(f"Output shape: {test_output.shape}")  # (2,2)

criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4)  # As recommended
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)  # Cosine decay example

def concordance_correlation_coefficient(y_true, y_pred):
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    covariance = np.cov(y_true, y_pred)[0,1]
    rho = covariance / np.sqrt(var_true * var_pred) if var_true * var_pred > 0 else 0
    ccc = (2 * rho * np.sqrt(var_true) * np.sqrt(var_pred)) / (var_true + var_pred + (mean_true - mean_pred)**2)
    return ccc

num_epochs = 10
best_val_loss = float('inf')

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for specs, labels in train_loader:
        specs = (specs - global_mean) / global_std  # Normalize
        specs, labels = specs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(specs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}")
    scheduler.step()
    
    model.eval()
    val_loss = 0
    val_preds, val_true = [], []
    with torch.no_grad():
        for specs, labels in val_loader:
            specs = (specs - global_mean) / global_std
            specs, labels = specs.to(device), labels.to(device)
            outputs = model(specs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            val_preds.append(outputs.cpu().numpy())
            val_true.append(labels.cpu().numpy())
    val_loss /= len(val_loader)
    val_preds = np.vstack(val_preds)
    val_true = np.vstack(val_true)
    mse_v = mean_squared_error(val_true[:,0], val_preds[:,0])
    mse_a = mean_squared_error(val_true[:,1], val_preds[:,1])
    ccc_v = concordance_correlation_coefficient(val_true[:,0], val_preds[:,0])
    ccc_a = concordance_correlation_coefficient(val_true[:,1], val_preds[:,1])
    print(f"Val Loss: {val_loss:.4f}, MSE V/A: {mse_v:.4f}/{mse_a:.4f}, CCC V/A: {ccc_v:.4f}/{ccc_a:.4f}")
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), '/kaggle/working/best_model.pth')

        model.load_state_dict(torch.load('/kaggle/working/best_model.pth'))
model.eval()
test_preds, test_true = [], []
with torch.no_grad():
    for specs, labels in test_loader:
        specs = (specs - global_mean) / global_std
        specs, labels = specs.to(device), labels.to(device)
        outputs = model(specs)
        test_preds.append(outputs.cpu().numpy())
        test_true.append(labels.cpu().numpy())
test_preds = np.vstack(test_preds)
test_true = np.vstack(test_true)

# Invert scaling
test_preds[:,0] = scaler_valence.inverse_transform(test_preds[:,0].reshape(-1,1)).flatten()
test_preds[:,1] = scaler_arousal.inverse_transform(test_preds[:,1].reshape(-1,1)).flatten()
test_true[:,0] = scaler_valence.inverse_transform(test_true[:,0].reshape(-1,1)).flatten()
test_true[:,1] = scaler_arousal.inverse_transform(test_true[:,1].reshape(-1,1)).flatten()

mse_v = mean_squared_error(test_true[:,0], test_preds[:,0])
mse_a = mean_squared_error(test_true[:,1], test_preds[:,1])
ccc_v = concordance_correlation_coefficient(test_true[:,0], test_preds[:,0])
ccc_a = concordance_correlation_coefficient(test_true[:,1], test_preds[:,1])
print(f"Test MSE V/A: {mse_v:.4f}/{mse_a:.4f}, CCC V/A: {ccc_v:.4f}/{ccc_a:.4f}")

# Same as before, but add norm
# ... (adapt from previous)
preds = []
with torch.no_grad():
    for segment in segments:
        # Compute mel_spec_db as before
        mel_spec_db = (torch.tensor(mel_spec_db).unsqueeze(0).unsqueeze(0) - global_mean) / global_std
        input_tensor = mel_spec_db.to(device)
        pred = model(input_tensor)
        preds.append(pred.cpu().numpy())
# Average and invert as before