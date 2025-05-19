import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification

# Cargar el modelo y el procesador
model_name = 'facebook/wav2vec2-base'
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)

# Función para diagnóstico
def diagnosticar_errores(audio_inputs):
    try:
        print("Iniciando diagnóstico...")

        # Verificar tipos y tamaños de entrada
        print("Tipo de entrada:", type(audio_inputs))
        print("Tamaño de la entrada:", len(audio_inputs))

        # Procesar entradas
        inputs = processor(audio_inputs, sampling_rate=16000, padding=True, return_tensors="pt")

        # Mostrar formas de tensores
        for key, value in inputs.items():
            print(f"Tensor {key} - forma: {value.shape} - dtype: {value.dtype}")

        # Pasar las entradas al modelo
        with torch.no_grad():
            outputs = model(**inputs)

        print("Diagnóstico completado: El modelo se ejecutó sin errores.")

    except Exception as e:
        print("Error detectado durante el diagnóstico:")
        print(str(e))


# Prueba de ejemplo
if __name__ == "__main__":
    # Audio de prueba
    test_audio = [torch.randn(80000) for _ in range(8)]  # Simula 8 entradas de audio
    diagnosticar_errores(test_audio)

