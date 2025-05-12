import os
import shutil
from sklearn.model_selection import train_test_split

def split_ravdess_dataset(source_dir, train_dir, val_dir, train_size=0.8, seed=42):
    """
    Split RAVDESS dataset into train and validation sets by actors.
    
    Args:
        source_dir (str): Path to Audio_Song_Actors_01-24 folder
        train_dir (str): Path to output train folder
        val_dir (str): Path to output validation folder
        train_size (float): Proportion of actors for training (default: 0.8)
        seed (int): Random seed for reproducibility
    """
    # Create output directories
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Get list of actor folders
    actor_folders = [f for f in os.listdir(source_dir) if f.startswith('Actor_')]
    
    # Split actors into train and validation
    train_actors, val_actors = train_test_split(
        actor_folders, train_size=train_size, random_state=seed
    )
    
    print(f"Train actors: {len(train_actors)} ({train_actors})")
    print(f"Validation actors: {len(val_actors)} ({val_actors})")
    
    # Function to copy speech files
    def copy_speech_files(actor_folder, dest_dir):
        actor_path = os.path.join(source_dir, actor_folder)
        for file in os.listdir(actor_path):
            if file.endswith('.wav') and file.split('-')[0] == '03':  # Speech files
                src_path = os.path.join(actor_path, file)
                dest_path = os.path.join(dest_dir, file)
                shutil.copy2(src_path, dest_path)
    
    # Copy files for train actors
    for actor in train_actors:
        copy_speech_files(actor, train_dir)
    
    # Copy files for validation actors
    for actor in val_actors:
        copy_speech_files(actor, val_dir)
    
    print(f"Train files: {len(os.listdir(train_dir))}")
    print(f"Validation files: {len(os.listdir(val_dir))}")

if __name__ == "__main__":
    SOURCE_DIR = "/emotion-classification/data/Audio_Song_Actors_01-24"
    TRAIN_DIR = "/emotion-classification/data/ravdess/train"
    VAL_DIR = "/emotion-classification/data/ravdess/val"
    
    split_ravdess_dataset(SOURCE_DIR, TRAIN_DIR, VAL_DIR)