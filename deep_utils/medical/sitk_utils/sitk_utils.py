from typing import Tuple, Union

import SimpleITK as sitk  # noqa
import numpy as np
from SimpleITK import Image


class SITKUtils:
    @staticmethod
    def get_array_img(sample_path: str, just_array: bool = False, just_sitk_img: bool = False) -> Union[
        Image, np.ndarray, Tuple[np.ndarray, Image]]:
        image = sitk.ReadImage(sample_path)
        array = sitk.GetArrayFromImage(image)
        if just_array and just_sitk_img:
            raise ValueError("Both just keywords cannot be set to True!")
        if just_array:
            return array
        if just_sitk_img:
            return image

        return array, image
