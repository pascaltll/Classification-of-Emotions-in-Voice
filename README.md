# Classification-of-Emotions-in-Voice
Classify emotions (e.g. happiness, sadness, anger) in voice recordings.

Project Overview

The goal is to classify eight emotions (neutral, calm, happy, sad, angry, fearful, surprise, disgust) from speech recordings. The project uses:


- Wav2Vec2: A transformer-based model from Hugging Face, fine-tuned for emotion classification.
- RAVDESS Dataset: Audio clips from 24 actors expressing emotions.
- Mel-Spectrograms: Audio is preprocessed into mel-spectrograms using torchaudio.
- Metrics: Accuracy and F1-score, visualized with wandb.



Tools: PyTorch, Hugging Face, Hydra (for configuration), wandb (for logging), and DVC (for data versioning).

The project fulfills course requirements by implementing a neural network for sequence processing, using modern frameworks, and comparing results with a baseline.