from deep_utils.dummy_objects import DummyObject, requires_backends


class YOLOV5TorchObjectDetector(metaclass=DummyObject):
    _backend = ["torch", "cv2", "seaborn", "pyaml"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, self._backend)
