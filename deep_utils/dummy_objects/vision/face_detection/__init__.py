from deep_utils.dummy_objects import DummyObject, requires_backends


class HaarcascadeCV2FaceDetector(metaclass=DummyObject):
    _backend = ["cv2"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, self._backend)


class MTCNNTFFaceDetector(metaclass=DummyObject):
    _backend = ["tensorflow", "cv2"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, self._backend)


class MTCNNTorchFaceDetector(metaclass=DummyObject):
    _backend = ["torch", "cv2"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, self._backend)


class RetinaFaceTorchFaceDetector(metaclass=DummyObject):
    _backend = ["torch", "cv2"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, self._backend)


class SSDCV2CaffeFaceDetector(metaclass=DummyObject):
    _backend = ["cv2"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, self._backend)


class UltralightTFFaceDetector(metaclass=DummyObject):
    _backend = ["tensorflow", "cv2"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, self._backend)


class UltralightTorchFaceDetector(metaclass=DummyObject):
    _backend = ["torch", "cv2"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, self._backend)
