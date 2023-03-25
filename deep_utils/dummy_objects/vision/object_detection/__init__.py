from deep_utils.dummy_objects.dummy_framework import DummyObject, requires_backends


class YOLOV5TorchObjectDetector(metaclass=DummyObject):
    _backend = ["torch", "cv2", "seaborn", "psutil", "yaml", "ipython"]
    _module = "deep_utils.vision.object_detection.yolo.v5.torch.yolo_v5_torch_object_detection"

    def __init__(self, *args, **kwargs):
        requires_backends(self, self._backend, module_name=self._module, cls_name=self.__class__.__name__)


class YOLOV7TorchObjectDetector(metaclass=DummyObject):
    _backend = ["torch", "cv2", "seaborn", "yaml"]
    _module = "deep_utils.vision.object_detection.yolo.v7.torch.yolo_v7_torch_object_detection"

    def __init__(self, *args, **kwargs):
        requires_backends(self, self._backend, module_name=self._module, cls_name=self.__class__.__name__)
