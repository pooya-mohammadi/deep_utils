from .main import ObjectDetector
from deep_utils.utils.lib_utils.main_utils import list_utils, loader
from deep_utils.utils.lib_utils.framework_utils import is_torch_available

# if not is_torch_available():
#     # Here go the dummy objects
#     from deep_utils.dummy_objects.vision.object_detection import YOLOV5TorchObjectDetector
# else:
#     from deep_utils.vision.object_detection.yolo.v5.torch.yolo_v5_torch_object_detection import \
#         YOLOV5TorchObjectDetector
try:
    from deep_utils.dummy_objects.vision.object_detection import YOLOV5TorchObjectDetector
    from deep_utils.vision.object_detection.yolo.v5.torch.yolo_v5_torch_object_detection import \
        YOLOV5TorchObjectDetector
except:
    pass

Object_Detection_Models = {
    "YOLOV5TorchObjectDetector": YOLOV5TorchObjectDetector
}

list_object_detection_models = list_utils(Object_Detection_Models)


def object_detector_loader(name, **kwargs) -> ObjectDetector:
    return loader(Object_Detection_Models, list_object_detection_models)(name, **kwargs)
