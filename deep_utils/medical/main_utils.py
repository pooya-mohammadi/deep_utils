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
        expands = [int(d * expand / 100) if expand else 0 for d in shape]

        if len(mins) == 3:
            cropped_array = array[
                            max(0, mins[0] - expands[0]): min(shape[0], maxs[0] + expands[0]),
                            max(0, mins[1] - expands[1]): min(shape[1], maxs[1] + expands[1]),
                            max(0, mins[2] - expands[2]): min(shape[2], maxs[2] + expands[2])]

        elif len(mins) == 2:
            cropped_array = array[
                            max(0, mins[0] - expands[0]): min(shape[0], maxs[0] + expands[0]),
                            max(0, mins[1] - expands[1]): min(shape[1], maxs[1] + expands[1])]
        elif len(mins) == 4:
            cropped_array = array[
                            max(0, mins[0] - expands[0]): min(shape[0], maxs[0] + expands[0]),
                            max(0, mins[1] - expands[1]): min(shape[1], maxs[1] + expands[1]),
                            max(0, mins[2] - expands[2]): min(shape[2], maxs[2] + expands[2]),
                            max(0, mins[3] - expands[3]): min(shape[3], maxs[3] + expands[3])
                            ]
        else:
            raise ValueError(f"Something wrong with mins: {mins} and maxs: {maxs}")
        if get_info:
            info = {i: (mi, ma) for i, (mi, ma) in enumerate(zip(mins, maxs))}
            return cropped_array, info
        else:
            return cropped_array

    @staticmethod
    def crop_with_info(array: np.ndarray, info: dict, expand: int = 0):
        mins, maxs = [], []
        for k, v in sorted(info.items(), key=lambda x: x[0]):
            if isinstance(k, int):
                mins.append(v[0])
                maxs.append(v[1])

        shape = array.shape
        expands = [int(d * expand / 100) if expand else 0 for d in shape]

        if len(mins) == 3:
            cropped_array = array[
                            max(0, mins[0] - expands[0]): min(shape[0], maxs[0] + expands[0]),
                            max(0, mins[1] - expands[1]): min(shape[1], maxs[1] + expands[1]),
                            max(0, mins[2] - expands[2]): min(shape[2], maxs[2] + expands[2])]

        elif len(mins) == 2:
            cropped_array = array[
                            max(0, mins[0] - expands[0]): min(shape[0], maxs[0] + expands[0]),
                            max(0, mins[1] - expands[1]): min(shape[1], maxs[1] + expands[1])]
        elif len(mins) == 4:
            cropped_array = array[
                            max(0, mins[0] - expands[0]): min(shape[0], maxs[0] + expands[0]),
                            max(0, mins[1] - expands[1]): min(shape[1], maxs[1] + expands[1]),
                            max(0, mins[2] - expands[2]): min(shape[2], maxs[2] + expands[2]),
                            max(0, mins[3] - expands[3]): min(shape[3], maxs[3] + expands[3])
                            ]
        else:
            raise ValueError(f"Something wrong with mins: {mins} and maxs: {maxs}")
        return            cropped_array