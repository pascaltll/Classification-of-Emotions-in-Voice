# +
import os
import hydra
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

@hydra.main(config_path="../configs", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    # Resolve relative paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg.data.train_dir = os.path.join(project_root, cfg.data.train_dir)

    # Collect all .wav files
    files = []
    labels = []
    for root, _, filenames in os.walk(cfg.data.train_dir):
        for file in filenames:
            if file.endswith('.wav'):
                files.append(os.path.join(root, file))
                emotion_id = int(file.split('-')[2]) - 1
                labels.append(emotion_id)

    if not files:
        raise ValueError(f"No .wav files found in {cfg.data.train_dir}")

    print(f"Total files found: {len(files)}")

    # Split the dataset (stratified by emotion)
    train_indices, val_indices = train_test_split(
        list(range(len(files))),
        test_size=0.2,  # 80-20 split
        random_state=cfg.training.seed,
        stratify=labels
    )

    # Define output directories
    train_split_dir = os.path.join(project_root, 'data/ravdess/train_split')
    val_split_dir = os.path.join(project_root, 'data/ravdess/val_split')
    os.makedirs(train_split_dir, exist_ok=True)
    os.makedirs(val_split_dir, exist_ok=True)

    # Move files to train split
    for idx in train_indices:
        src = files[idx]
        relative_path = os.path.relpath(src, cfg.data.train_dir)
        dst = os.path.join(train_split_dir, relative_path)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        os.rename(src, dst)

    # Move files to validation split
    for idx in val_indices:
        src = files[idx]
        relative_path = os.path.relpath(src, cfg.data.train_dir)
        dst = os.path.join(val_split_dir, relative_path)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        os.rename(src, dst)

    print(f"Train split size: {len(train_indices)} files moved to {train_split_dir}")
    print(f"Validation split size: {len(val_indices)} files moved to {val_split_dir}")

if __name__ == "__main__":
    main()
