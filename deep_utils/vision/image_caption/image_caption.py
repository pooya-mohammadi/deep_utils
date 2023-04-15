from typing import Tuple, Union
from abc import abstractmethod, ABCMeta

import numpy as np


class ImageCaption(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self, weight_path: str, device: str, img_w: int = None, img_h: int = None,
                 img_norm_mean: Tuple[float, float, float] = None, img_norm_std: Tuple[float, float, float] = None,
                 **params):
        self._weight_path = weight_path
        self._device = device
        self._img_w, self._img_h = img_w, img_h
        self._img_norm_mean = img_norm_mean
        self._img_norm_std = img_norm_std

    @abstractmethod
    def generate_caption(self, image: Union[np.array, str]) -> str:
        raise NotImplementedError
