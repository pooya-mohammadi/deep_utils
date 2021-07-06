from .main import *
from .ssd import *
from deep_utils.vision.face_detection.main import FaceDetector

Face_Detection_Models = {
    "SSDCV2CaffeFaceDetector": SSDCV2CaffeFaceDetector
}


def list_face_detection_models():
    detection_models = ""
    for name, _ in Face_Detection_Models.items():
        detection_models += f'{name}\n'
    print(detection_models)


def face_detector_loader(name, *args, **kwargs) -> FaceDetector:
    if name not in Face_Detection_Models:
        raise Exception(f'{name} model is not supported. Supported models are {list_face_detection_models()}')
    model = Face_Detection_Models[name](*args, **kwargs)
    return model
