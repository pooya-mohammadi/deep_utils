from deep_utils.main_abs.dummy_framework.dummy_framework import DummyObject, requires_backends


class YOLOV5TorchObjectDetector(metaclass=DummyObject):
    _backend = ["torch", "cv2", "seaborn", "pyaml"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, self._backend)


class YOLOV7TorchObjectDetector(metaclass=DummyObject):
    _backend = ["torch", "cv2", "seaborn", "pyaml"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, self._backend)
