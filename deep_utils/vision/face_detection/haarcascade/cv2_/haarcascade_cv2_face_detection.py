from deep_utils.utils.box_utils.boxes import Box, Point
from deep_utils.utils.lib_utils.download_utils import download_decorator
from deep_utils.utils.lib_utils.lib_decorators import (
    expand_input,
    get_elapsed_time,
    get_from_config,
    rgb2bgr,
)
from deep_utils.vision.face_detection.main import FaceDetector

from .config import Config


class HaarcascadeCV2FaceDetector(FaceDetector):
    def __init__(self, **kwargs):
        super().__init__(
            name=self.__class__.__name__,
            file_path=__file__,
            download_variables=(
                "haarcascade_frontalface",
                "haarcascade_eye",
                "haarcascade_nose",
                "landmarks",
            ),
            **kwargs
        )
        self.config: Config

    @download_decorator
    def load_model(self):
        import cv2

        face_detector = cv2.CascadeClassifier(
            self.config.haarcascade_frontalface)
        nose_detector = cv2.CascadeClassifier(self.config.haarcascade_nose)
        eye_detector = cv2.CascadeClassifier(self.config.haarcascade_eye)
        landmark_detector = cv2.face.createFacemarkLBF()
        landmark_detector.loadModel(self.config.landmarks)
        self.model = dict(
            face=face_detector,
            nose=nose_detector,
            eye=eye_detector,
            landmarks=landmark_detector,
        )

    @get_elapsed_time
    @expand_input(3)
    @get_from_config
    @rgb2bgr("gray")
    def detect_faces(
        self,
        img,
        is_rgb,
        scaleFactor=None,
        minNeighbors=None,
        minSize=None,
        maxSize=None,
        flags=None,
        get_landmarks=True,
        get_nose=True,
        get_eye=True,
        get_time=False,
    ):
        boxes, landmarks_, confidences, eye_poses, nose_poses = [], [], [], [], []
        for image in img:
            faces = self.model["face"].detectMultiScale(
                image,
                scaleFactor=scaleFactor,
                minNeighbors=minNeighbors,
                minSize=minSize,
                maxSize=maxSize,
                flags=flags,
            )
            boxes.append(
                Box.box2box(
                    faces,
                    in_source="CV",
                    to_source="Numpy",
                    in_format="XYWH",
                    to_format="XYXY",
                )
            )
            if get_nose:
                nose_pos = self.model["nose"].detectMultiScale(
                    image,
                    scaleFactor=scaleFactor,
                    minNeighbors=minNeighbors,
                    minSize=minSize,
                    maxSize=maxSize,
                    flags=flags,
                )
                nose_pos = Point.point2point(
                    nose_pos, in_source="CV", to_source="Numpy"
                )
                nose_poses.append(nose_pos)
            if get_eye:
                eye_pos = self.model["eye"].detectMultiScale(
                    image,
                    scaleFactor=scaleFactor,
                    minNeighbors=minNeighbors,
                    minSize=minSize,
                    maxSize=maxSize,
                    flags=flags,
                )
                eye_pos = Point.point2point(
                    eye_pos, in_source="CV", to_source="Numpy")
                eye_poses.append(eye_pos)
            if len(faces) != 0 and get_landmarks:
                _, landmarks = self.model["landmarks"].fit(image, faces)
                landmarks = [
                    Point.point2point(
                        face_landmarks[0].tolist(), in_source="CV", to_source="Numpy"
                    )
                    for face_landmarks in landmarks
                ]
                landmarks_.append(landmarks)
        return dict(
            boxes=boxes,
            confidences=confidences,
            landmarks=landmarks_,
            eye_poses=eye_poses,
            nose_poses=nose_poses,
        )
