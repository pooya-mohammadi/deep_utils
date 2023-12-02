from typing import Tuple, Union, Dict, List, Optional

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
    def swap_seg_value(input_array: np.ndarray, swap_seg: Dict[int, int], max_val: int = 256) -> np.ndarray:
        """

        :param input_array:
        :param swap_seg:
        :param max_val:
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

        def random_val(values: List[Union[int, float]], max_val_: int = 256) -> float:
            import random
            val = random.randint(max(values + [1]), max_val_)
            while val in values:
                val = random.random()
            return val

        unique_values = np.unique(input_array).tolist()
        reverse_swap = dict()
        for source, target in swap_seg.items():
            source = reverse_swap.pop(source, source)
            random_target = random_val(unique_values + list(reverse_swap.keys()), max_val_=max_val)
            input_array[input_array == target] = random_target
            input_array[input_array == source] = target
            reverse_swap[target] = random_target

        for target, source in reverse_swap.items():
            input_array[input_array == source] = target
        return input_array

    @staticmethod
    def swap_seg_value_file(input_file: str,
                            swap_seg: Dict[int, int],
                            output_file: str,
                            **kwarg):
        """
        Swap class value over a sample!
        """
        array, image = SITKUtils.get_array_img(input_file)
        swaped_array = SITKUtils.swap_seg_value(array, swap_seg)
        SITKUtils.save_sample(swaped_array, image, output_file, **kwarg)

    @staticmethod
    def save_sample(input_sample, org_sitk_img, save_path: str, time_array_index=-1,
                    direction: Optional[list] = None, spacing: Optional[list] = None):
        """

        :param input_sample:
        :param org_sitk_img:
        :param save_path:
        :param time_array_index: This is for 4D data
        :param direction:
        :param spacing: if provided will be used in the save
        :return:
        """
        slices = []
        if len(input_sample.shape) == 4:
            for t in range(input_sample.shape[time_array_index]):
                sample_sitk = sitk.GetImageFromArray(input_sample[..., t] if time_array_index else input_sample[t],
                                                     False)
                slices.append(sample_sitk)
            sample_sitk = sitk.JoinSeries(slices)
            org_origin = org_sitk_img.GetOrigin()
            sample_sitk.SetOrigin((*org_origin, 1.0) if len(org_origin) == 3 else org_origin)
            org_spacing = org_sitk_img.GetSpacing()
            sample_sitk.SetSpacing((*org_spacing, 1.0) if len(org_spacing) == 3 else org_spacing)
            org_direction = np.array(org_sitk_img.GetDirection())
            if org_direction.size == 9:
                org_direction = org_direction.reshape(3, 3)
                org_direction = np.pad(org_direction, [(0, 1), (0, 1)], mode='constant', constant_values=1).flatten()
            sample_sitk.SetDirection(org_direction)
        else:
            sample_sitk = sitk.GetImageFromArray(input_sample, False)
            # sample_sitk.SetOrigin(org_sitk.GetOrigin())
            # sample_sitk.SetSpacing(org_sitk.GetSpacing())
            sample_sitk.SetOrigin(org_sitk_img.GetOrigin()[:3])
            if spacing is not None:
                sample_sitk.SetSpacing(spacing)
            else:
                sample_sitk.SetSpacing(org_sitk_img.GetSpacing()[:3])

            # Extract the 3x3 sub-matrix from the original 4x4 direction matrix
            if direction is not None:
                sample_sitk.SetDirection(np.array(direction).flatten())
            else:
                org_flat_direction = np.array(org_sitk_img.GetDirection())
                if len(org_flat_direction) == 9:
                    original_direction = org_flat_direction.reshape(3, 3)
                elif len(org_flat_direction) == 16:
                    original_direction = org_flat_direction.reshape(4, 4)
                else:
                    raise ValueError()
                submatrix_direction = original_direction[:3, :3].flatten()
                sample_sitk.SetDirection(submatrix_direction)
        sitk.WriteImage(sample_sitk, save_path)

    @staticmethod
    def update_file(file_path: str, target_path: Optional[str] = None,
                    spacing_x: Optional[float] = None,
                    spacing_y: Optional[float] = None,
                    spacing_z: Optional[float] = None,
                    spacing_t: Optional[float] = None
                    ):
        target_path = target_path or file_path
        if not spacing_x and not spacing_y and not spacing_z and not spacing_t:
            raise ValueError("spacing cannot be empty for all dimensions")

        array, img = SITKUtils.get_array_img(file_path)
        spacing = list(img.GetSpacing())
        spacing[0] = spacing_x or spacing[0]
        spacing[1] = spacing_y or spacing[1]
        spacing[2] = spacing_z or spacing[2]
        if len(spacing) > 3:
            spacing[3] = spacing_t or spacing[3]
        SITKUtils.save_sample(array, img, spacing=spacing, save_path=target_path)


if __name__ == '__main__':
    array = np.array([[1, 3, 1, 2, 1], [1, 3, 2, 2, 1]])
    output = SITKUtils.swap_seg_value(array, swap_seg={1: 3, 3: 1})
    print(output)
