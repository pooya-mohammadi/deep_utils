from deep_utils.dummy_objects import DummyObject, requires_backends


class MemoryUtilsTorch(metaclass=DummyObject):
    _backend = ["torch"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, self._backend)