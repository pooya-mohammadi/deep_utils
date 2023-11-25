from deep_utils.dummy_objects.dummy_framework import DummyObject, requires_backends


class PyannoteAudioDiarization(metaclass=DummyObject):
    _backend = ["pyannote", "torchaudio"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, self._backend)
