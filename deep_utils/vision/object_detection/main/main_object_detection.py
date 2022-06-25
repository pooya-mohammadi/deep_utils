from abc import abstractmethod
from typing import Union

import numpy as np

from deep_utils.main_abs.main import MainClass


class ObjectDetector(MainClass):
    def __init__(self, name, file_path, *args, **kwargs):
        super().__init__(name, file_path=file_path, **kwargs)
        self.confidences: Union[list, None, np.ndarray] = None
        self.boxes: Union[list, None, np.ndarray] = None
        self.classes: Union[list, None, np.ndarray] = None
        self.class_names: Union[list, None, np.ndarray] = None

    @abstractmethod
    def detect_objects(
        self,
        img,
        is_rgb,
        confidence=None,
        iou_thresh=None,
        classes=None,
        get_time=False,
        **kwargs
    ):
        pass

    @abstractmethod
    def detect_dir(
        self,
        dir_,
        confidence=None,
        iou_thresh=None,
        classes=None,
        extensions=(".png", ".jpg", ".jpeg"),
        save_results=False,
        save_in_file=False,
        **kwargs
    ):
        pass
