from deep_utils.dummy_objects.dummy_framework import DummyObject, requires_backends


class TorchVisionInference(metaclass=DummyObject):
    _backend = ["torch", "cv2", "torchvision"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, self._backend)


class TorchVisionModel(metaclass=DummyObject):
    _backend = ["torch", "numpy", "torchvision"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, self._backend)
