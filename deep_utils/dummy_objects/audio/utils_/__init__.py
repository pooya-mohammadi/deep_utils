from deep_utils.main_abs.dummy_framework.dummy_framework import DummyObject, requires_backends


class TorchAudioUtils(metaclass=DummyObject):
    _backend = ["torch", "torchaudio"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, self._backend)
