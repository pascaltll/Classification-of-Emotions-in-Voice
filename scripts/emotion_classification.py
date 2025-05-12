import torch
import torchaudio
import librosa
import wandb
import hydra
from omegaconf import DictConfig
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor, Trainer, TrainingArguments
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import os

# Custom Dataset for RAVDESS
class RAVDESSDataset(Dataset):
    def __init__(self, data_dir, processor, max_length=5.0, sample_rate=16000):
        self.data_dir = data_dir
        self.processor = processor
        self.max_length = max_length
        self.sample_rate = sample_rate
        self.emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'surprise', 'disgust']
        self.files = []
        self.labels = []
        
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.wav'):
                    self.files.append(os.path.join(root, file))
                    emotion_id = int(file.split('-')[2]) - 1
                    self.labels.append(emotion_id)
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file_path = self.files[idx]
        label = self.labels[idx]
        
        waveform, sr = torchaudio.load(file_path)
        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)
        
        max_samples = int(self.max_length * self.sample_rate)
        if waveform.shape[1] > max_samples:
            waveform = waveform[:, :max_samples]
        else:
            padding = max_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        
        inputs = self.processor(waveform.squeeze().numpy(), sampling_rate=self.sample_rate, return_tensors="pt", padding=True)
        return {
            'input_values': inputs.input_values.squeeze(),
            'attention_mask': inputs.attention_mask.squeeze(),
            'labels': label
        }

# Compute metrics for evaluation
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    return {'accuracy': acc, 'f1': f1}

@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # Initialize wandb
    wandb.init(project="emotion-classification", config=dict(cfg))
    
    # Load processor and model
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        "facebook/wav2vec2-base", num_labels=8
    )
    
    # Load datasets
    train_dataset = RAVDESSDataset(cfg.data.train_dir, processor)
    val_dataset = RAVDESSDataset(cfg.data.val_dir, processor)
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=cfg.model.save_path,
        evaluation_strategy="epoch",
        learning_rate=cfg.training.lr,
        per_device_train_batch_size=cfg.training.batch_size,
        per_device_eval_batch_size=cfg.training.batch_size,
        num_train_epochs=cfg.training.epochs,
        logging_dir='./logs',
        logging_steps=10,
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="wandb",
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    # Train
    trainer.train()
    
    # Save model
    trainer.save_model(cfg.model.save_path)
    processor.save_pretrained(cfg.model.save_path)

if __name__ == "__main__":
    main()