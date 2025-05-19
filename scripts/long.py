import torchaudio
import os
import numpy as np
from omegaconf import DictConfig
import hydra

def analyze_audio_lengths(data_dir, sample_rate=16000):
    """
    Analiza la longitud de los archivos de audio en un directorio.

    Args:
        data_dir (str): Directorio que contiene los archivos de audio .wav.
        sample_rate (int): Frecuencia de muestreo esperada (para calcular la duración).

    Returns:
        tuple: (max_duration, average_duration) en segundos.
    """
    durations = []
    filenames = [f for f in os.listdir(data_dir) if f.endswith('.wav')]
    if not filenames:
        print(f"No se encontraron archivos .wav en el directorio: {data_dir}")
        return 0, 0

    for filename in filenames:
        file_path = os.path.join(data_dir, filename)
        try:
            waveform, sr = torchaudio.load(file_path)
            if sr != sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
                waveform = resampler(waveform)
            duration = waveform.shape[-1] / sample_rate
            durations.append(duration)
        except Exception as e:
            print(f"Error al cargar o procesar {filename}: {e}")

    if not durations:
        return 0, 0

    max_duration = np.max(durations)
    average_duration = np.mean(durations)
    return max_duration, average_duration

@hydra.main(config_path="../configs", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    train_dir = cfg.data.train_dir
    val_dir = cfg.data.val_dir
    sample_rate = cfg.data.sample_rate

    print(f"Analizando el conjunto de entrenamiento en: {train_dir}")
    max_train_duration, avg_train_duration = analyze_audio_lengths(train_dir, sample_rate)
    print(f"Longitud máxima en el conjunto de entrenamiento: {max_train_duration:.2f} segundos")
    print(f"Longitud promedio en el conjunto de entrenamiento: {avg_train_duration:.2f} segundos")

    print(f"\nAnalizando el conjunto de validación en: {val_dir}")
    max_val_duration, avg_val_duration = analyze_audio_lengths(val_dir, sample_rate)
    print(f"Longitud máxima en el conjunto de validación: {max_val_duration:.2f} segundos")
    print(f"Longitud promedio en el conjunto de validación: {avg_val_duration:.2f} segundos")

    overall_max_duration = max(max_train_duration, max_val_duration)
    overall_avg_duration = np.mean([avg_train_duration, avg_val_duration])

    print("\nAnálisis general:")
    print(f"Longitud máxima global: {overall_max_duration:.2f} segundos")
    print(f"Longitud promedio global: {overall_avg_duration:.2f} segundos")

    suggested_max_length = overall_max_duration  # Puedes ajustar esto según tus necesidades
    print(f"\nLongitud máxima sugerida para cortar/padding (basada en el máximo): {suggested_max_length:.2f} segundos")
    print(f"Podrías considerar un valor cercano al promedio ({overall_avg_duration:.2f} segundos) si quieres evitar demasiado padding, pero ten en cuenta la posible pérdida de información.")

if __name__ == "__main__":
    main()
