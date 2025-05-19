import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import librosa
import numpy as np
import os
import wandb
import hydra
from omegaconf import DictConfig
from sklearn.metrics import accuracy_score, f1_score

# CNN Model Definition
class CNNAudioClassifier(nn.Module):
    def __init__(self, input_shape=(13, 100), num_classes=8):
        super(CNNAudioClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * (input_shape[0] // 8) * (input_shape[1] // 8), 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Dataset Class with MFCCs and Data Augmentation
class RAVDESSDataset(Dataset):
    def __init__(self, data_dir, max_length=5.0, sample_rate=16000, n_mfcc=13):
        self.data_dir = data_dir
        self.max_length = max_length
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprise']
        self.files = []
        self.labels = []

        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.wav'):
                    self.files.append(os.path.join(root, file))
                    emotion_id = int(file.split('-')[2]) - 1
                    self.labels.append(emotion_id)

        if not self.files:
            raise ValueError(f"No .wav files found in {data_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        label = self.labels[idx]

        # Load audio
        signal, sr = torchaudio.load(file_path)
        if sr != self.sample_rate:
            signal = torchaudio.transforms.Resample(sr, self.sample_rate)(signal).squeeze()

        # Trim or pad to max_length
        max_samples = int(self.max_length * self.sample_rate)
        if signal.shape[-1] > max_samples:
            signal = signal[:, :max_samples]
        else:
            padding = max_samples - signal.shape[-1]
            signal = torch.nn.functional.pad(signal, (0, padding))

        # Convert to numpy and apply data augmentation (e.g., add noise)
        signal = signal.numpy().squeeze()
        if np.random.random() > 0.7:  # 30% chance of augmentation
            noise = np.random.normal(0, 0.005, signal.shape)
            signal = signal + noise

        # Extract MFCCs
        mfcc = librosa.feature.mfcc(y=signal, sr=self.sample_rate, n_mfcc=self.n_mfcc, hop_length=512)
        mfcc = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)  # Add channel dimension

        return {
            'input_values': mfcc,
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Training Function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_preds, train_labels = [], []

        for batch in train_loader:
            inputs = batch['input_values'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        # Validation
        model.eval()
        val_loss = 0
        val_preds, val_labels = [], []

        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['input_values'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        # Compute metrics
        train_acc = accuracy_score(train_labels, train_preds)
        val_acc = accuracy_score(val_labels, val_preds)
        train_f1 = f1_score(train_labels, train_preds, average='weighted')
        val_f1 = f1_score(val_labels, val_preds, average='weighted')

        # Log to WandB
        wandb.log({
            'epoch': epoch + 1,
            'train_loss': train_loss / len(train_loader),
            'val_loss': val_loss / len(val_loader),
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'train_f1': train_f1,
            'val_f1': val_f1
        })

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.4f}')

@hydra.main(config_path="../configs", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    # Resolve relative paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg.data.train_dir = os.path.join(project_root, cfg.data.train_dir)
    cfg.data.val_dir = os.path.join(project_root, cfg.data.val_dir)

    # Initialize WandB
    wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity, config=dict(cfg))

    # Load datasets
   
