class Config:
    device = 'cpu'
    min_face_size = 20.0
    thresholds = (0.6, 0.7, 0.8)
    nms_thresholds = (0.7, 0.7, 0.7)
    min_detection_size = 12
    factor = 0.707
    confidence = 0.8
    pnet_url = "https://github.com/Practical-AI/deep_utils/releases/download/0.1.0/pnet.h5"
    pnet_cache = 'weights/vision/face_detection/mtcnn/tf/pnet.h5'
    pnet = None
    rnet_url = "https://github.com/Practical-AI/deep_utils/releases/download/0.1.0/rnet.h5"
    rnet_cache = 'weights/vision/face_detection/mtcnn/tf/rnet.h5'
    rnet = None
    onet_url = "https://github.com/Practical-AI/deep_utils/releases/download/0.1.0/onet.h5"
    onet_cache = 'weights/vision/face_detection/mtcnn/tf/onet.h5'
    onet = None
