import numpy as np
from typing import Literal


class NumpyUtils:
    @staticmethod
    def combine_without_overlap(left_array: np.ndarray, right_array: np.ndarray, merge_type: Literal['left', "right"]):
        """

        :param left_array:
        :param right_array:
        :param merge_type:
        :return:
        >>> left = np.array([[1, 1], [0, 0]])
        >>> right = np.array([[0, 2], [2, 0]])
        >>> NumpyUtils.combine_without_overlap(left, right, "left")
        array([[1, 1],
               [2, 0]])
        >>> NumpyUtils.combine_without_overlap(left, right, "right")
        array([[1, 2],
               [2, 0]])
        """
        output_array = np.sum(np.array([left_array, right_array]), axis=0)
        left_ones = np.zeros_like(left_array)
        left_ones[left_array > 0] = 1
        right_ones = np.zeros_like(right_array)
        right_ones[right_array > 0] = 1
        overlap = np.where(left_ones == right_ones)
        if merge_type == 'left':
            output_array[overlap] = left_array[overlap]
        elif merge_type == 'right':
            output_array[overlap] = right_array[overlap]
        else:
            raise ValueError(f"merge_type: {merge_type} is not supported!")

        return output_array
