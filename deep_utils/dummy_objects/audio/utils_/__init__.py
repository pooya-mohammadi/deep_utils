from deep_utils.dummy_objects.dummy_framework import DummyObject, requires_backends


class TorchAudioUtils(metaclass=DummyObject):
    _backend = ["torch", "torchaudio"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, self._backend)


class LibrosaUtils(metaclass=DummyObject):
    _backend = ["numpy", "librosa"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, self._backend)
