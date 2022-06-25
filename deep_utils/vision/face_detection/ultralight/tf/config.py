class Config:
    def __init__(self):
        self.RBF_url = (
            "https://github.com/Practical-AI/deep_utils/releases/download/0.2.0/RBF.zip"
        )
        self.RBF_cache = "weights/vision/face_detection/ultra-light/tf/RBF/RBF.zip"
        self.RBF = None
        self.slim_url = "https://github.com/Practical-AI/deep_utils/releases/download/0.2.0/slim.zip"
        self.slim_cache = "weights/vision/face_detection/ultra-light/tf/slim/slim.zip"
        self.slim = None
