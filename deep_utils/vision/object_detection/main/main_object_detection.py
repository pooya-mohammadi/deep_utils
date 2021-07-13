from typing import Union
from abc import abstractmethod
import numpy as np
from deep_utils.main_abs.main import MainClass


class ObjectDetector(MainClass):
    def __init__(self, name, file_path, **kwargs):
        super().__init__(name, file_path=file_path, **kwargs)
        self.confidences: Union[list, None, np.ndarray] = None
        self.boxes: Union[list, None, np.ndarray] = None
        self.classes: Union[list, None, np.ndarray] = None
        self.class_names: Union[list, None, np.ndarray] = None

    @abstractmethod
    def detect_objects(self, img, is_rgb, confidence=None, iou_thresh=None, get_time=False):
        pass
