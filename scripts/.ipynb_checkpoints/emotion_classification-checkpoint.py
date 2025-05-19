import torch
import torchaudio
import wandb
import hydra
from omegaconf import DictConfig
from transformers import (Wav2Vec2ForSequenceClassification, Wav2Vec2Processor, 
                          Trainer, TrainingArguments, DataCollatorWithPadding)
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import os

class RAVDESSDataset(Dataset):
    def __init__(self, data_dir, processor, max_length=5.0, sample_rate=16000):
        self.data_dir = data_dir
        self.processor = processor
        self.max_length = int(max_length * sample_rate)  
        self.sample_rate = sample_rate
        self.files = [f for f in os.listdir(data_dir) if f.endswith('.wav')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.files[idx])
        label = int(self.files[idx].split('-')[2]) - 1

        waveform, sr = torchaudio.load(file_path)
        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)

        waveform = waveform.squeeze().numpy()

        inputs = self.processor(waveform, sampling_rate=self.sample_rate, return_tensors="pt", 
                                padding="max_length", max_length=self.max_length, truncation=True)

        return {'input_values': inputs.input_values.squeeze(), 'labels': label}


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {'accuracy': accuracy_score(labels, predictions), 'f1': f1_score(labels, predictions, average='weighted')}


@hydra.main(config_path="../configs", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/wav2vec2-base", num_labels=8)

    train_dataset = RAVDESSDataset(cfg.data.train_dir, processor, cfg.data.max_length, cfg.data.sample_rate)
    val_dataset = RAVDESSDataset(cfg.data.val_dir, processor, cfg.data.max_length, cfg.data.sample_rate)

    data_collator = DataCollatorWithPadding(
        processor, 
        padding="longest",  # Cambia a "longest" para asegurar que las secuencias se alineen correctamente
        return_tensors="pt"
    )

    model.gradient_checkpointing_enable()

    training_args = TrainingArguments(
        output_dir=cfg.model.save_path,
        evaluation_strategy="epoch",
        learning_rate=cfg.training.lr,
        per_device_train_batch_size=cfg.training.batch_size,
        per_device_eval_batch_size=cfg.training.batch_size,
        num_train_epochs=cfg.training.epochs,
        logging_dir='./logs',
        save_strategy="epoch",
        load_best_model_at_end=True,
        gradient_checkpointing=True,
        seed=cfg.training.seed,
        report_to="wandb"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(cfg.model.save_path)
    processor.save_pretrained(cfg.model.save_path)


if __name__ == "__main__":
    main()

