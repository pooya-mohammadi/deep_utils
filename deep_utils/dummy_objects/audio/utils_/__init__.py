from deep_utils.dummy_objects import DummyObject, requires_backends


class TorchAudioUtils(metaclass=DummyObject):
    _backend = ["torch", "torchaudio"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, self._backend)
