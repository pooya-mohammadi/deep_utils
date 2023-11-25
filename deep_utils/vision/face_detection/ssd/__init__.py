try:
    from deep_utils.dummy_objects.vision.face_detection import SSDCV2CaffeFaceDetector
    from .cv2.caffe.ssd_cv2_caffe_face_detection import SSDCV2CaffeFaceDetector
except ModuleNotFoundError:
    pass
