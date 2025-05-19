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

```
- Download `Audio_Song_Actors_01-24.zip` from [Zenodo](https://zenodo.org/record/1188976).

- Unzip to `/emotion-classification/data/Audio_Song_Actors_01-24`.
```

Filename identifiers 

Modality (01 = full-AV, 02 = video-only, 03 = audio-only).
Vocal channel (01 = speech, 02 = song).
Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the 'neutral' emotion.
Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").
Repetition (01 = 1st repetition, 02 = 2nd repetition).
Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).

Filename example: 02-01-06-01-02-01-12.mp4 

Video-only (02)
Speech (01)
Fearful (06)
Normal intensity (01)
Statement "dogs" (02)
1st Repetition (01)
12th Actor (12)
Female, as the actor ID number is even.
