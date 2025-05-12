# Classification-of-Emotions-in-Voice
Classify emotions (e.g. happiness, sadness, anger) in voice recordings.

Project Overview

The goal is to classify eight emotions (neutral, calm, happy, sad, angry, fearful, surprise, disgust) from speech recordings. The project uses:

- Wav2Vec2: A transformer-based model from Hugging Face, fine-tuned for emotion classification.
- CNN Baseline: A convolutional neural network using mel-spectrograms as input.
- RAVDESS Dataset: Audio clips from 24 actors expressing emotions.
- Mel-Spectrograms: Audio is preprocessed into mel-spectrograms using torchaudio and visualized with librosa.
- Metrics: Accuracy, F1-score, and confusion matrix, visualized with wandb and matplotlib.
- Tools: PyTorch, Hugging Face, torchaudio, Librosa, Hydra, wandb, DVC, git.





data set:
https://zenodo.org/records/1188976