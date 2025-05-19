import os
import pandas as pd
import torch
import hydra
from datasets import Dataset, Audio
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import wandb
import numpy as np
from omegaconf import DictConfig

# Hydra configuration loading
@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Set random seed for reproducibility
    torch.manual_seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)

    # Initialize WandB
    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        config=dict(cfg),  # Log all config parameters
    )

    # Data paths from config
    train_dir = cfg.data.train_dir
    val_dir = cfg.data.val_dir
    sample_rate = cfg.data.sample_rate
    max_length_seconds = cfg.data.max_length

    # Map of IDs to emotions
    id2emotion = {
        '01': 'neutral',
        '02': 'calm',
        '03': 'happy',
        '04': 'sad',
        '05': 'angry',
        '06': 'fearful',
        '07': 'disgust',
        '08': 'surprise'
    }

    # Function to load data from directory
    def load_data(data_dir):
        files = [f for f in os.listdir(data_dir) if f.endswith('.wav')]
        data = []
        for f in files:
            parts = f.split('-')
            emotion_id = parts[2]
            emotion = id2emotion.get(emotion_id)
            if emotion:
                data.append({
                    'path': os.path.join(data_dir, f),
                    'label': emotion
                })
        return pd.DataFrame(data)

    # Load train and validation datasets
    train_df = load_data(train_dir)
    val_df = load_data(val_dir)

    # Convert to Hugging Face Datasets
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    # Cast audio column
    train_dataset = train_dataset.cast_column("path", Audio(sampling_rate=sample_rate))
    val_dataset = val_dataset.cast_column("path", Audio(sampling_rate=sample_rate))

    # Load processor
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")

    # Preprocess function with max_length
    def preprocess(batch):
        audio = batch["path"]["array"]
        # Calculate max samples based on max_length_seconds
        max_samples = int(sample_rate * max_length_seconds)
        if len(audio) > max_samples:
            audio = audio[:max_samples]
        inputs = processor(
            audio,
            sampling_rate=sample_rate,
            padding=True,
            return_tensors=None
        )
        batch["input_values"] = inputs["input_values"][0]  # np.array of floats
        return batch

    # Apply preprocessing
    train_dataset = train_dataset.map(preprocess)
    val_dataset = val_dataset.map(preprocess)

    # Create label mappings
    labels = sorted(list(set(train_df['label'])))
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}

    # Map labels to IDs
    def label_to_id(batch):
        batch["labels"] = label2id[batch["label"]]
        return batch

    train_dataset = train_dataset.map(label_to_id)
    val_dataset = val_dataset.map(label_to_id)

    # Remove unnecessary columns
    train_dataset = train_dataset.map(preprocess, remove_columns=["path", "label"])
    val_dataset = val_dataset.map(preprocess, remove_columns=["path", "label"])

    # Load model
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        "facebook/wav2vec2-base",
        num_labels=len(labels),
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True
    )

    # Training arguments from config
    training_args = TrainingArguments(
        output_dir=cfg.model.save_path,
        per_device_train_batch_size=cfg.training.batch_size,
        per_device_eval_batch_size=cfg.training.batch_size,
        eval_strategy="steps",
        eval_steps=500,
        save_steps=500,
        logging_steps=cfg.wandb.log_freq,
        learning_rate=cfg.training.lr,
        num_train_epochs=cfg.training.epochs,
        save_total_limit=2,
        metric_for_best_model="accuracy",
        load_best_model_at_end=True,
        seed=cfg.training.seed,
        report_to="wandb",  # Enable WandB logging
    )

    # Compute metrics
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

    # Data collator
    def data_collator(features):
        input_values = [f["input_values"] for f in features]
        labels = [f["labels"] for f in features]
        batch = processor.pad(
            {"input_values": input_values},
            padding=True,
            return_tensors="pt"
        )
        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        return batch

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=processor.feature_extractor,
        compute_metrics=compute_metrics,
    )

    # Train the model
    trainer.train()

    # Save the model
    trainer.save_model(cfg.model.save_path)
    processor.save_pretrained(cfg.model.save_path)

    # Finish WandB run
    wandb.finish()

if __name__ == "__main__":
    main()
