from deep_utils.dummy_objects.dummy_framework import DummyObject, requires_backends


class AudioClassificationWav2vec2Torch(metaclass=DummyObject):
    _backend = ["torch", "librosa", "transformers", "torchaudio"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, self._backend)
