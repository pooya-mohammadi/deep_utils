class Config:
    def __init__(self):
        self.device = 'cpu'
        self.RBF_url = "https://github.com/Practical-AI/deep_utils/releases/download/0.2.0/version-RFB-320.pth"
        self.RBF_cache = 'weights/vision/face_detection/ultra-light/torch/RBF/version-RFB-320.pth'
        self.RBF = None
        self.slim_url = "https://github.com/Practical-AI/deep_utils/releases/download/0.2.0/version-slim-320.pth"
        self.slim_cache = 'weights/vision/face_detection/ultra-light/torch/slim/version-slim-320.pth'
        self.slim = None
