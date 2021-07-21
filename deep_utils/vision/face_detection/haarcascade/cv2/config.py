class Config:
    haarcascade_frontalface_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt2.xml"
    haarcascade_frontalface_cache = "weights/vision/face_detection/haarcascade/cv2/haarcascade_frontalface_alt2.xml"
    haarcascade_frontalface = None
    haarcascade_eye_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml"
    haarcascade_eye_cache = "weights/vision/face_detection/haarcascade/cv2/haarcascade_eye.xml"
    haarcascade_eye = None
    haarcascade_nose_url = "https://github.com/Practical-AI/deep_utils/releases/download/0.1.0/haarcascade_nose.xml"
    haarcascade_nose_cache = "weights/vision/face_detection/haarcascade/cv2/haarcascade_nose.xml"
    haarcascade_nose = None
    landmarks_url = "https://github.com/kurnianggoro/GSOC2017/raw/master/data/lbfmodel.yaml"
    landmarks_cache = "weights/vision/face_detection/haarcascade/cv2/lbfmodel.yaml"
    landmarks = None

    scale = None
    scaleFactor = None
    minSize = None
    minNeighbors = None
    flags = None
    maxSize = None
