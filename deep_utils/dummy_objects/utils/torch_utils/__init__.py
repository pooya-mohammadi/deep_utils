from deep_utils.main_abs.dummy_framework.dummy_framework import DummyObject, requires_backends


class TorchUtils(metaclass=DummyObject):
    _backend = ["torch"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, self._backend)
