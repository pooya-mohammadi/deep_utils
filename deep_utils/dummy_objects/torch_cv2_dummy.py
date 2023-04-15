from deep_utils.dummy_objects.dummy_framework import DummyObject, requires_backends


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
