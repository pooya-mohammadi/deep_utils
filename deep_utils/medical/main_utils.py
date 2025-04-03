import numpy as np
from typing import List, Tuple


class MainMedUtils:
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
