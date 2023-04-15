try:
    from deep_utils._dummy_objects.audio.classification.wav2vec2 import AudioClassificationWav2vec2Torch
    from .audio_classification_wav2vec2 import AudioClassificationWav2vec2Torch
except ModuleNotFoundError:
    pass
