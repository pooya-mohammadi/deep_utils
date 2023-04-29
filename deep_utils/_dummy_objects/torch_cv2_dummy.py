from deep_utils._dummy_objects.dummy_framework import DummyObject, requires_backends


class MTCNNTorchFaceDetector(metaclass=DummyObject):
    _backend = ["torch", "cv2"]
    _module = "deep_utils.vision.face_detection.mtcnn.torch.mtcnn_torch_face_detection"

    def __init__(self, *args, **kwargs):
        requires_backends(self, self._backend, module_name=self._module, cls_name=self.__class__.__name__)


class YOLOV5TorchObjectDetector(metaclass=DummyObject):
    _backend = ["torch", "cv2"]
    _module = "deep_utils.vision.object_detection.yolo.v5.torch.yolo_v5_torch_object_detection"

    def __init__(self, *args, **kwargs):
        requires_backends(self, self._backend, module_name=self._module, cls_name=self.__class__.__name__)


class YOLOV7TorchObjectDetector(metaclass=DummyObject):
    _backend = ["torch", "cv2"]
    _module = "deep_utils.vision.object_detection.yolo.v7.torch.yolo_v7_torch_object_detection"

    def __init__(self, *args, **kwargs):
        requires_backends(self, self._backend, module_name=self._module, cls_name=self.__class__.__name__)


class ColorRecognitionCNNTorchPrediction(metaclass=DummyObject):
    _backend = [("torch", "1.13.1", "pip"),
                ("opencv-python", " 4.7.0.68", "pip")]
    _module = "deep_utils.vision.color_recognition.cnn_color.torch.color_cnn_torch_pred"

    def __init__(self, *args, **kwargs):
        requires_backends(self, self._backend, module_name=self._module, cls_name=self.__class__.__name__)
