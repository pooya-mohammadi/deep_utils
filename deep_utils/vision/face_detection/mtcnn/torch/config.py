class Config:
    device = 'cpu'
    min_face_size = 20.0
    thresholds = (0.6, 0.7, 0.8)
    nms_thresholds = (0.7, 0.7, 0.7)
    min_detection_size = 12
    factor = 0.707
    pnet_url = "https://github.com/Practical-AI/deep_utils/releases/download/0.1.0/pnet.npy"
    pnet_cache = 'weights/vision/face_detection/mtcnn/torch/pnet.npy'
    pnet = None
    rnet_url = "https://github.com/Practical-AI/deep_utils/releases/download/0.1.0/rnet.npy"
    rnet_cache = 'weights/vision/face_detection/mtcnn/torch/rnet.npy'
    rnet = None
    onet_url = "https://github.com/Practical-AI/deep_utils/releases/download/0.1.0/onet.npy"
    onet_cache = 'weights/vision/face_detection/mtcnn/torch/onet.npy'
    onet = None
