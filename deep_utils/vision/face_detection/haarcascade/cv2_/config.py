class Config:
    def __init__(self):
        self.haarcascade_frontalface_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt2.xml"
        self.haarcascade_frontalface_cache = "weights/vision/face_detection/haarcascade/cv2/haarcascade_frontalface_alt2.xml"
        self.haarcascade_frontalface = None
        self.haarcascade_eye_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml"
        self.haarcascade_eye_cache = (
            "weights/vision/face_detection/haarcascade/cv2/haarcascade_eye.xml"
        )
        self.haarcascade_eye = None
        self.haarcascade_nose_url = "https://github.com/Practical-AI/deep_utils/releases/download/0.1.0/haarcascade_nose.xml"
        self.haarcascade_nose_cache = (
            "weights/vision/face_detection/haarcascade/cv2/haarcascade_nose.xml"
        )
        self.haarcascade_nose = None
        self.landmarks_url = (
            "https://github.com/kurnianggoro/GSOC2017/raw/master/data/lbfmodel.yaml"
        )
        self.landmarks_cache = (
            "weights/vision/face_detection/haarcascade/cv2/lbfmodel.yaml"
        )
        self.landmarks = None

        self.scale = None
        self.scaleFactor = None
        self.minSize = None
        self.minNeighbors = None
        self.flags = None
        self.maxSize = None
        self.download_variables = ("haarcascade_frontalface", "haarcascade_eye", "haarcascade_nose", "landmarks")
