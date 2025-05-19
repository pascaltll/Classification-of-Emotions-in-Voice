import torch
import torchaudio
import wandb
import hydra
from omegaconf import DictConfig
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor, Trainer, TrainingArguments
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import os

class RAVDESSDataset(Dataset):
    def __init__(self, data_dir, processor, max_length=5.0, sample_rate=16000):
        self.data_dir = data_dir
        self.processor = processor
        self.max_length = max_length
        self.sample_rate = sample_rate
        self.emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'surprise', 'disgust']
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

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    return {'accuracy': acc, 'f1': f1}

@hydra.main(config_path="../configs", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    # Resolve relative paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg.data.train_dir = os.path.join(project_root, cfg.data.train_dir)
    cfg.data.val_dir = os.path.join(project_root, cfg.data.val_dir)
    cfg.model.save_path = os.path.join(project_root, cfg.model.save_path)
    #################3
    print(f"Train dir: {cfg.data.train_dir}")
    print(f"Val dir: {cfg.data.val_dir}")
    print(f"Train dir exists: {os.path.exists(cfg.data.train_dir)}")
    print(f"Train dir contents: {os.listdir(cfg.data.train_dir) if os.path.exists(cfg.data.train_dir) else 'Directory not found'}")
    ####################
    # wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity, config=dict(cfg))
    
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        "facebook/wav2vec2-base", num_labels=8
    )
    
    train_dataset = RAVDESSDataset(cfg.data.train_dir, processor, cfg.data.max_length, cfg.data.sample_rate)
    val_dataset = RAVDESSDataset(cfg.data.val_dir, processor, cfg.data.max_length, cfg.data.sample_rate)
    
    training_args = TrainingArguments(
        output_dir=cfg.model.save_path,
        evaluation_strategy="epoch",
        learning_rate=cfg.training.lr,
        per_device_train_batch_size=cfg.training.batch_size,
        per_device_eval_batch_size=cfg.training.batch_size,
        num_train_epochs=cfg.training.epochs,
        logging_dir=os.path.join(project_root, 'logs'),
        logging_steps=cfg.wandb.log_freq,
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none",  # Desactiva wandb
        seed=cfg.training.seed,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    trainer.train()
    
    trainer.save_model(cfg.model.save_path)
    processor.save_pretrained(cfg.model.save_path)

if __name__ == "__main__":
    main()
