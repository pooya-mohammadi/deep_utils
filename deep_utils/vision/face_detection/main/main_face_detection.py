from typing import Union
from abc import abstractmethod
import numpy as np
from deep_utils.main_abs.main import MainClass


class FaceDetector(MainClass):
    def __init__(self, name, file_path, **kwargs):
        super().__init__(name, file_path=file_path, **kwargs)

    @abstractmethod
    def detect_faces(self, img, is_rgb, confidence=None, get_time=False):
        pass
