# from deep_utils.utils.lib_utils.main_utils import list_utils, loader
# from .haarcascade import *
# from .mtcnn import *
# from .retinaface import *
# from .ssd import *
# from .ultralight import *
# from .main import *
#
# Face_Detection_Models = {
#     "SSDCV2CaffeFaceDetector": SSDCV2CaffeFaceDetector,
#     "MTCNNTorchFaceDetector": MTCNNTorchFaceDetector,
#     "HaarcascadeCV2FaceDetector": HaarcascadeCV2FaceDetector,
#     "MTCNNTFFaceDetector": MTCNNTFFaceDetector,
#     "UltralightTorchFaceDetector": UltralightTorchFaceDetector,
#     # "UltralightTFFaceDetector": UltralightTFFaceDetector,
#     "RetinaFaceTorchFaceDetector": RetinaFaceTorchFaceDetector,
# }
#
# list_face_detection_models = list_utils(Face_Detection_Models)
#
#
# def face_detector_loader(name, **kwargs) -> FaceDetector:
#     return loader(Face_Detection_Models, list_face_detection_models)(name, **kwargs)
