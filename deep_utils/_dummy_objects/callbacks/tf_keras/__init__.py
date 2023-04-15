from deep_utils._dummy_objects.dummy_framework import DummyObject, requires_backends


class LRScalar(metaclass=DummyObject):
    _backend = ["tensorflow"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, self._backend)
