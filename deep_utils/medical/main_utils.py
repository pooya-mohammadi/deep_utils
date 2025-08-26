import random
import numpy as np
from typing import List, Tuple, Union


class MainMedUtils:
    @staticmethod
    def crop_pad(crop, image, seg=None):
        if seg is not None:
            assert seg.shape == image.shape

        if len(image.shape) == 3:
            i_z, i_y, i_x = image.shape
            c_z, c_y, c_x = crop
            start_crop_z = random.randint(0, i_z - c_z)
            start_crop_y = random.randint(0, i_y - c_y)
            start_crop_x = random.randint(0, i_x - c_x)

            end_crop_z = start_crop_z + c_z
            end_crop_y = start_crop_y + c_y
            end_crop_x = start_crop_x + c_x

            image = image[start_crop_z: end_crop_z, start_crop_y: end_crop_y, start_crop_x: end_crop_x]
            if seg is not None:
                seg = seg[start_crop_z: end_crop_z, start_crop_y: end_crop_y, start_crop_x: end_crop_x]
                return image, seg
            return image

    @staticmethod
    def get_largets_box(array: np.ndarray, get_info: bool = False):
        dimensions = np.where(array)
        mins = []
        maxs = []
        for dim in dimensions:
            if len(dim):
                mins.append(int(np.min(dim)))
                maxs.append(int(np.max(dim)))
            else:
                mins.append(None)
                maxs.append(None)
        if get_info:
            info = {i: (mi, ma) for i, (mi, ma) in enumerate(zip(mins, maxs))}
            return info
        return mins, maxs

    @staticmethod
    def compute_new_shape(old_shape: Union[Tuple[int, ...], List[int], np.ndarray],
                          old_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
                          new_spacing: Union[Tuple[float, ...], List[float], np.ndarray]) -> np.ndarray:
        assert len(old_spacing) == len(old_shape), "Shapes are not equal"
        assert len(old_shape) == len(new_spacing), "Shapes are not equal"

        new_shape = np.array(
            [int(round(old_spacing_ / new_spacing_ * old_shape_)) for old_spacing_, new_spacing_, old_shape_ in
             zip(old_spacing, new_spacing, old_shape)])

        return new_shape

    @staticmethod
    def get_largest_box_and_crop(array: np.ndarray, expand: int = 0, get_info: bool = False):
        mins, maxs = MainMedUtils.get_largets_box(array)
        shape = array.shape
        expands = np.array([int(d * expand / 100) if expand else 0 for d in shape])
        mins = np.maximum(np.array(mins) - expands, 0)
        maxs = np.minimum(np.array(maxs) + expands, shape)
        if len(mins) == 3:
            cropped_array = array[
                mins[0]: maxs[0],
                mins[1]: maxs[1],
                mins[2]: maxs[2]
            ]

        elif len(mins) == 2:
            cropped_array = array[
                mins[0]: maxs[0],
                mins[1]: maxs[1]
            ]
        elif len(mins) == 4:
            cropped_array = array[
                mins[0]: maxs[0],
                mins[1]: maxs[1],
                mins[2]: maxs[2],
                mins[3]: maxs[3],
            ]
        else:
            raise ValueError(f"Something wrong with mins: {mins} and maxs: {maxs}")

        if get_info:
            info = {i: (int(mi), int(ma)) for i, (mi, ma) in enumerate(zip(mins, maxs))}
            return cropped_array, info
        else:
            return cropped_array

    @staticmethod
    def crop_with_info(array: np.ndarray, info: dict):
        mins, maxs = [], []
        # expand = info.pop('expand')
        for k, v in sorted([(k, v) for k, v in info.items() if isinstance(k, int) or k.isdigit()], key=lambda x: x[0]):
            mins.append(v[0])
            maxs.append(v[1])

        if len(mins) == 3:
            cropped_array = array[
                mins[0]: maxs[0],
                mins[1]: maxs[1],
                mins[2]: maxs[2]
            ]

        elif len(mins) == 2:
            cropped_array = array[
                mins[0]: maxs[0],
                mins[1]: maxs[1]
            ]
        elif len(mins) == 4:
            cropped_array = array[
                mins[0]: maxs[0],
                mins[1]: maxs[1],
                mins[2]: maxs[2],
                mins[3]: maxs[3],
            ]
        else:
            raise ValueError(f"Something wrong with mins: {mins} and maxs: {maxs}")
        return cropped_array
