from typing import Union
from abc import abstractmethod
import numpy as np
from deep_utils.main_abs.main import MainClass


class FaceDetector(MainClass):
    def __init__(self, name, file_path, **kwargs):
        super().__init__(name, file_path=file_path, **kwargs)
        self.model = None
        self.confidences: Union[list, None, np.ndarray] = None
        self.boxes: Union[list, None, np.ndarray] = None
        self.landmarks: Union[list, None, np.ndarray] = None

    @abstractmethod
    def detect_faces(self, img, confidence=None, return_landmark=False, get_time=False):
        pass
