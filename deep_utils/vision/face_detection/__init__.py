from .main import *
from .ssd import *
from .mtcnn import *
from .haarcascade import *
from .ultralight import *
from .retinaface import *
from deep_utils.utils.lib_utils.main_utils import list_utils, loader

Face_Detection_Models = {
    "SSDCV2CaffeFaceDetector": SSDCV2CaffeFaceDetector,
    "MTCNNTorchFaceDetector": MTCNNTorchFaceDetector,
    "HaarcascadeCV2FaceDetector": HaarcascadeCV2FaceDetector,
    "MTCNNTFFaceDetector": MTCNNTFFaceDetector,
    "UltralightTorchFaceDetector": UltralightTorchFaceDetector,
    "UltralightTFFaceDetector": UltralightTFFaceDetector,
    "RetinaFaceTorchFaceDetector": RetinaFaceTorchFaceDetector,
}

list_face_detection_models = list_utils(Face_Detection_Models)


def face_detector_loader(name, **kwargs) -> FaceDetector:
    return loader(Face_Detection_Models, list_face_detection_models)(name, **kwargs)
