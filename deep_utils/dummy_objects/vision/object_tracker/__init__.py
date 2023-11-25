from deep_utils.dummy_objects.dummy_framework import DummyObject, requires_backends


class DeepSortTorch(metaclass=DummyObject):
    _backend = ["torch", "cv2"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, self._backend)


class DeepSortTorchTracker(metaclass=DummyObject):
    _backend = ["torch", "cv2"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, self._backend)


class DeepSortTorchFeatureExtractor(metaclass=DummyObject):
    _backend = ["torch", "cv2"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, self._backend)
