from deep_utils.main_abs.cv2.cv2_config import CV2Config


class Config(CV2Config):
    prototxt_url = 'https://github.com/Practical-AI/deep_utils/releases/download/0.1.0/deploy.prototxt.txt'
    prototxt_cache = 'weights/vision/face_detection/sdd/caffe/deploy.prototxt.txt'
    prototxt = None
    caffemodel_url = 'https://github.com/Practical-AI/deep_utils/releases/download/0.1.0/res10_300x300_ssd_iter_140000.caffemodel'
    caffemodel_cache = 'weights/vision/face_detection/sdd/caffe/res10_300x300_ssd_iter_140000.caffemodel'
    caffemodel = None
    confidence = 0.5
