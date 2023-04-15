from deep_utils.dummy_objects.dummy_framework import DummyObject, requires_backends


class HaarcascadeCV2FaceDetector(metaclass=DummyObject):
    _backend = ["cv2"]
    _module = "deep_utils.vision.face_detection.haarcascade.cv2_.haarcascade_cv2_face_detection"

    def __init__(self, *args, **kwargs):
        requires_backends(self, self._backend, module_name=self._module, cls_name=self.__class__.__name__)


class CVUtils(metaclass=DummyObject):
    _backend = ["cv2"]
    _module = "deep_utils.utils.opencv_utils.opencv_utils"

    def __init__(self, *args, **kwargs):
        requires_backends(self, self._backend, module_name=self._module, cls_name=self.__class__.__name__)
