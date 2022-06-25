from deep_utils.dummy_objects import DummyObject, requires_backends


class VggFace2TorchFaceRecognition(metaclass=DummyObject):
    _backend = ["torch", "cv2", "albumentation", "scikit-learn"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, self._backend)
