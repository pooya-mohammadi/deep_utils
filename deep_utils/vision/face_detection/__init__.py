from .main import *
from .ssd import *
from deep_utils.utils.lib_utils.main_utils import list_utils, loader

Face_Detection_Models = {
    "SSDCV2CaffeFaceDetector": SSDCV2CaffeFaceDetector
}

list_face_detection_models = list_utils(Face_Detection_Models)


def face_detector_loader(name, **kwargs) -> FaceDetector:
    return loader(Face_Detection_Models, list_face_detection_models)(name, **kwargs)
