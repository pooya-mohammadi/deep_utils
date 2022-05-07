from .main import FaceDetector
from deep_utils.utils.lib_utils.main_utils import list_utils, loader
from deep_utils.utils.lib_utils.framework_utils import is_torch_available, is_cv2_available, is_tf_available

if not is_torch_available():
    from deep_utils.dummy_objects.vision.face_detection import MTCNNTorchFaceDetector
    from deep_utils.dummy_objects.vision.face_detection import UltralightTorchFaceDetector
    from deep_utils.dummy_objects.vision.face_detection import RetinaFaceTorchFaceDetector
else:
    from deep_utils.vision.face_detection.mtcnn.torch.mtcnn_torch_face_detection import MTCNNTorchFaceDetector
    from deep_utils.vision.face_detection.ultralight.torch.ultralight_torch_face_detection import \
        UltralightTorchFaceDetector
    from deep_utils.vision.face_detection.retinaface.torch.retinaface_torch_face_detection import \
        RetinaFaceTorchFaceDetector

if not is_cv2_available():
    from deep_utils.dummy_objects.vision.face_detection import SSDCV2CaffeFaceDetector
    from deep_utils.dummy_objects.vision.face_detection import HaarcascadeCV2FaceDetector
else:
    from deep_utils.vision.face_detection.ssd.cv2.caffe.ssd_cv2_caffe_face_detection import SSDCV2CaffeFaceDetector
    from deep_utils.vision.face_detection.haarcascade.cv2_.haarcascade_cv2_face_detection import \
        HaarcascadeCV2FaceDetector

if not is_tf_available():
    from deep_utils.dummy_objects.vision.face_detection import MTCNNTFFaceDetector
    from deep_utils.dummy_objects.vision.face_detection import UltralightTFFaceDetector
else:
    from deep_utils.vision.face_detection.mtcnn.tf.mtcnn_tf_face_detection import MTCNNTFFaceDetector
    from deep_utils.vision.face_detection.ultralight.tf.ultralight_tf_face_detection import UltralightTFFaceDetector


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
