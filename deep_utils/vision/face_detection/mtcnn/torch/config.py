from deep_utils.main_abs.main_config import MainConfig


class Config(MainConfig):
    def __init__(self):
        self.device = "cpu"
        self.min_face_size = 20.0
        self.thresholds = (0.6, 0.7, 0.8)
        self.nms_thresholds = (0.7, 0.7, 0.7)
        self.min_detection_size = 12
        self.factor = 0.707
        self.confidence = 0.8
        self.pnet_url = "https://github.com/pooya-mohammadi/deep_utils/releases/download/0.1.0/pnet.npy"
        self.pnet_cache = "weights/vision/face_detection/mtcnn/torch/pnet.npy"
        self.pnet = None
        self.rnet_url = "https://github.com/pooya-mohammadi/deep_utils/releases/download/0.1.0/rnet.npy"
        self.rnet_cache = "weights/vision/face_detection/mtcnn/torch/rnet.npy"
        self.rnet = None
        self.onet_url = "https://github.com/pooya-mohammadi/deep_utils/releases/download/0.1.0/onet.npy"
        self.onet_cache = "weights/vision/face_detection/mtcnn/torch/onet.npy"
        self.onet = None
        self.download_variables = ("pnet", "onet", "rnet")
