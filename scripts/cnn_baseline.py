import torch
import torch.nn as nn
import torchaudio
import librosa
import numpy as np
from torch.utils.data import Dataset, DataLoader
import wandb
import hydra
from omegaconf import DictConfig
from sklearn.metrics import accuracy_score, f1_score
import os

class CNN(nn.Module):
    def __init__(self, num_classes=8):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 32 * 25, 128)  # Adjust based on spectrogram size
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class RAVDESSMelDataset(Dataset):
    def __init__(self, data_dir, sample_rate=16000, n_mels=128, max_length=5.0):
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.max_length = max_length
        self.files = []
        self.labels = []
        
        for file in os.listdir(data_dir):
            if file.endswith('.wav'):
                self.files.append(os.path.join(data_dir, file))
                emotion_id = int(file.split('-')[2]) - 1
                self.labels.append(emotion_id)
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file_path = self.files[idx]
        label = self.labels[idx]
        
        signal, sr = librosa.load(file_path, sr=self.sample_rate)
        max_samples = int(self.max_length * self.sample_rate)
        if len(signal) > max_samples:
            signal = signal[:max_samples]
        else:
            signal = np.pad(signal, (0, max_samples - len(signal)), 'constant')
        
        mel_spec = librosa.feature.melspectrogram(y=signal, sr=self.sample_rate, n_mels=self.n_mels)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        mel_spec_db = torch.tensor(mel_spec_db, dtype=torch.float32).unsqueeze(0)
        return {
            'input': mel_spec_db,
            'label': torch.tensor(label, dtype=torch.long)
        }

def train_cnn(model, train_loader, val_loader, cfg, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    
    for epoch in range(cfg.training.epochs):
        model.train()
        train_loss, train_preds, train_labels = [], [], []
        for batch in train_loader:
            inputs = batch['input'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            train_loss.append(loss.item())
            train_preds.extend(outputs.argmax(dim=-1).cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
        
        model.eval()
        val_loss, val_preds, val_labels = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['input'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss.append(loss.item())
                val_preds.extend(outputs.argmax(dim=-1).cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        train_acc = accuracy_score(train_labels, train_preds)
        val_acc = accuracy_score(val_labels, val_preds)
        train_f1 = f1_score(train_labels, train_preds, average='weighted')
        val_f1 = f1_score(val_labels, val_preds, average='weighted')
        
        wandb.log({
            'epoch': epoch,
            'train_loss': np.mean(train_loss),
            'val_loss': np.mean(val_loss),
            'train_acc': train_acc,
            'val_acc': val_acc,
            'train_f1': train_f1,
            'val_f1': val_f1
        })
        
        print(f"Epoch {epoch}: Train Loss={np.mean(train_loss):.4f}, Val Loss={np.mean(val_loss):.4f}, "
              f"Val Acc={val_acc:.4f}, Val F1={val_f1:.4f}")

@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig):
    wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity, config=dict(cfg))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = CNN(num_classes=8)
    train_dataset = RAVDESSMelDataset(
        cfg.data.train_dir, cfg.data.sample_rate, cfg.data.n_mels, cfg.data.max_length
    )
    val_dataset = RAVDESSMelDataset(
        cfg.data.val_dir, cfg.data.sample_rate, cfg.data.n_mels, cfg.data.max_length
    )
    train_loader = DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.training.batch_size)
    
    train_cnn(model, train_loader, val_loader, cfg, device)
    
    torch.save(model.state_dict(), os.path.join(cfg.model.cnn_save_path, 'cnn_model.pth'))

if __name__ == "__main__":
    main()