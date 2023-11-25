import sys
from typing import Tuple, Union, Dict, List

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

    @staticmethod
    def swap_seg_value(input_array: np.ndarray, swap_seg: Dict[int, int]) -> np.ndarray:
        """

        :param input_array:
        :param swap_seg:
        :return:

        >>> array = np.array([[1, 3, 1, 2, 1], [1, 3, 2, 2, 1]])
        >>> SITKUtils.swap_seg_value(array, swap_seg={1: 3, 3: 1})
        array([[3, 1, 3, 2, 3],
               [3, 1, 2, 2, 3]])

        >>> array = np.array([[1, 3, 1, 2, 1], [1, 3, 2, 2, 1]])
        >>> SITKUtils.swap_seg_value(array, swap_seg={1: 3, 3: 1, 2: 1})
        array([[3, 1, 3, 1, 3],
               [3, 1, 1, 1, 3]])
        """

        def random_val(values: List[Union[int, float]]) -> float:
            import random
            val = random.randint(max(values + [1]), sys.maxsize)
            while val in values:
                val = random.random()
            return val

        unique_values = np.unique(input_array).tolist()
        reverse_swap = dict()
        for source, target in swap_seg.items():
            source = reverse_swap.pop(source, source)
            random_target = random_val(unique_values + list(reverse_swap.keys()))
            input_array[input_array == target] = random_target
            input_array[input_array == source] = target
            reverse_swap[target] = random_target

        for target, source in reverse_swap.items():
            input_array[input_array == source] = target
        return input_array

    @staticmethod
    def save_sample(sample, org_sitk, save_path: str, time_array_index=-1):
        """
        Since we use three dimension, we should get the origin and spacing of the first 3 dimensions
        """
        slices = []
        if len(sample.shape) == 4:
            for t in range(sample.shape[time_array_index]):
                sample_sitk = sitk.GetImageFromArray(sample[..., t] if time_array_index else sample[t], False)
                slices.append(sample_sitk)
            sample_sitk = sitk.JoinSeries(slices)
            org_origin = org_sitk.GetOrigin()
            sample_sitk.SetOrigin((*org_origin, 1.0) if len(org_origin) == 3 else org_origin)
            org_spacing = org_sitk.GetSpacing()
            sample_sitk.SetSpacing((*org_spacing, 1.0) if len(org_spacing) == 3 else org_spacing)
            org_direction = np.array(org_sitk.GetDirection())
            if org_direction.size == 9:
                org_direction = org_direction.reshape(3, 3)
                org_direction = np.pad(org_direction, [(0, 1), (0, 1)], mode='constant', constant_values=1).flatten()
            sample_sitk.SetDirection(org_direction)
        else:
            sample_sitk = sitk.GetImageFromArray(sample, False)
            # sample_sitk.SetOrigin(org_sitk.GetOrigin())
            # sample_sitk.SetSpacing(org_sitk.GetSpacing())
            sample_sitk.SetOrigin(org_sitk.GetOrigin()[:3])
            sample_sitk.SetSpacing(org_sitk.GetSpacing()[:3])

            # Extract the 3x3 sub-matrix from the original 4x4 direction matrix
            org_flat_direction = np.array(org_sitk.GetDirection())
            if len(org_flat_direction) == 9:
                original_direction = org_flat_direction.reshape(3, 3)
            elif len(org_flat_direction) == 16:
                original_direction = org_flat_direction.reshape(4, 4)
            else:
                raise ValueError()
            submatrix_direction = original_direction[:3, :3].flatten()
            sample_sitk.SetDirection(submatrix_direction)
        sitk.WriteImage(sample_sitk, save_path)


if __name__ == '__main__':
    array = np.array([[1, 3, 1, 2, 1], [1, 3, 2, 2, 1]])
    output = SITKUtils.swap_seg_value(array, swap_seg={1: 3, 3: 1})
    print(output)
