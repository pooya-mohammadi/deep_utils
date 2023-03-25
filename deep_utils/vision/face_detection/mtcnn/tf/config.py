class Config:
    def __init__(self):
        self.device = "cpu"
        self.min_face_size = 20.0
        self.thresholds = (0.6, 0.7, 0.8)
        self.nms_thresholds = (0.7, 0.7, 0.7)
        self.min_detection_size = 12
        self.factor = 0.707
        self.confidence = 0.8
        self.pnet_url = (
            "https://github.com/Practical-AI/deep_utils/releases/download/0.1.0/pnet.h5"
        )
        self.pnet_cache = "weights/vision/face_detection/mtcnn/tf/pnet.h5"
        self.pnet = None
        self.rnet_url = (
            "https://github.com/Practical-AI/deep_utils/releases/download/0.1.0/rnet.h5"
        )
        self.rnet_cache = "weights/vision/face_detection/mtcnn/tf/rnet.h5"
        self.rnet = None
        self.onet_url = (
            "https://github.com/Practical-AI/deep_utils/releases/download/0.1.0/onet.h5"
        )
        self.onet_cache = "weights/vision/face_detection/mtcnn/tf/onet.h5"
        self.onet = None
        self.download_variables = ("pnet", "onet", "rnet")
