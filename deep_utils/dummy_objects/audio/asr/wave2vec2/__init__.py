from deep_utils.dummy_objects import DummyObject, requires_backends


class Wav2Vec2STTTorch(metaclass=DummyObject):
    _backend = ["torch", "librosa", "transformers"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, self._backend)
