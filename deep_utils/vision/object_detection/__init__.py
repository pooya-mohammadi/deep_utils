# from deep_utils.utils.lib_utils.main_utils import list_utils, loader
#
# from .main import ObjectDetector
# from .yolo import *
#
# Object_Detection_Models = {"YOLOV5TorchObjectDetector": YOLOV5TorchObjectDetector,
#                            "YOLOV7TorchObjectDetector": YOLOV7TorchObjectDetector}
#
# list_object_detection_models = list_utils(Object_Detection_Models)
#
#
# def object_detector_loader(name, **kwargs) -> ObjectDetector:
#     return loader(Object_Detection_Models, list_object_detection_models)(name, **kwargs)
