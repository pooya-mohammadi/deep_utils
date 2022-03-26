from deep_utils.dummy_objects import DummyObject, requires_backends


class DeepSortTorch(metaclass=DummyObject):
    _backend = ["torch", "cv2"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, self._backend)
