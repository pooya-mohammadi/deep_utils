import math
from typing import Tuple, Union, Dict, List, Optional

import SimpleITK as sitk  # noqa
import numpy as np
from SimpleITK import Image


class SITKUtils:
    @staticmethod
    def get_array_img_properties(filepath: str):
        arr, img = SITKUtils.get_array_img(filepath)
        spacing = img.GetSpacing()
        origin = img.GetOrigin()
        direction = img.GetDirection()
        return arr, dict(spacing=spacing, origin=origin, direction=direction)

    @staticmethod
    def get_array(sample_path: str) -> np.ndarray:
        image = sitk.ReadImage(sample_path)
        array = sitk.GetArrayFromImage(image)
        return array

    @staticmethod
    def get_img(sample_path: str) -> Image:
        image = sitk.ReadImage(sample_path)
        return image

    @staticmethod
    def get_array_img(sample_path: str) -> Tuple[np.ndarray, Image]:
        image = sitk.ReadImage(sample_path)
        arr = sitk.GetArrayFromImage(image)
        return arr, image

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
    def write(sitk_img, save_path: str):
        """
        write sitk image!
        :param sitk_img:
        :param save_path:
        :return:
        """
        sitk.WriteImage(sitk_img, save_path)

    @staticmethod
    def save_sample_with_img(filepath: str, sample_array: np.ndarray, sitk_img: sitk.Image):
        img_ = sitk.GetImageFromArray(sample_array)
        img_.CopyInformation(sitk_img)
        img = sitk.Cast(img_, sitk_img.GetPixelID())
        sitk.WriteImage(sitk_img, filepath)

    @staticmethod
    def save_sample(input_sample: np.ndarray, org_sitk_img: Optional[Image],
                    save_path: str, time_array_index=-1,
                    direction: Optional[list] = None,
                    spacing: Optional[list] = None,
                    origin: Optional[list] = None,
                    remove_index: int = None,
                    slice_index: int = None):
        """

        :param input_sample:
        :param org_sitk_img:
        :param save_path:
        :param time_array_index: This is for 4D data
        :param direction:
        :param spacing: if provided will be used in the save
        :param origin:
        :param remove_index:
        :param slice_index:
        :return:
        """
        slices = []
        if len(input_sample.shape) == 4:
            for t in range(input_sample.shape[time_array_index]):
                sample_sitk = sitk.GetImageFromArray(input_sample[..., t] if time_array_index else input_sample[t],
                                                     False)
                slices.append(sample_sitk)
            sample_sitk = sitk.JoinSeries(slices)
            if org_sitk_img is None:
                org_sitk_img = sample_sitk

            if origin is None:
                org_origin = org_sitk_img.GetOrigin()
                sample_sitk.SetOrigin((*org_origin, 1.0) if len(org_origin) == 3 else org_origin)
            else:
                sample_sitk.SetOrigin(tuple(origin))
            org_spacing = org_sitk_img.GetSpacing()
            if len(org_spacing) == 5:
                if remove_index is None:
                    raise ValueError("remove index should be provided for 5 samples")
                org_spacing = list(org_spacing)
                del org_spacing[remove_index]
                org_spacing = tuple(org_spacing)
            else:
                org_spacing = (*org_spacing, 1.0) if len(org_spacing) == 3 else org_spacing
            if spacing:
                sample_sitk.SetSpacing(spacing)
            else:
                sample_sitk.SetSpacing(org_spacing)
            org_direction = np.array(org_sitk_img.GetDirection())
            if org_direction.size == 9:
                org_direction = org_direction.reshape(3, 3)
                org_direction = np.pad(org_direction, [(0, 1), (0, 1)], mode='constant', constant_values=1).flatten()
            if org_direction.size == 25:
                org_direction = org_direction.reshape((5, 5))
                if remove_index is None:
                    raise ValueError("remove index should be provided for 5,5 samples")
                org_direction = np.delete(org_direction, remove_index, 0)
                org_direction = np.delete(org_direction, remove_index, 1)
                org_direction = org_direction.flatten()

            try:
                sample_sitk.SetDirection(org_direction)
            except:
                print("[WARNING] Couldn't set the direction. Skipping....")
        elif len(input_sample.shape) == 3:
            sample_sitk = sitk.GetImageFromArray(input_sample, False)
            if org_sitk_img is None:
                org_sitk_img = sample_sitk
            if spacing is not None:
                sample_sitk.SetSpacing(spacing)
            else:
                spacing = list(org_sitk_img.GetSpacing())
                if remove_index is not None and len(spacing) > 3:
                    del spacing[remove_index]
                if slice_index is not None and len(spacing) > 3:
                    del spacing[slice_index]
                sample_sitk.SetSpacing(spacing)

            # Extract the 3x3 sub-matrix from the original 4x4 direction matrix
            if direction is not None:
                sample_sitk.SetDirection(np.array(direction).flatten())
            else:
                org_flat_direction = np.array(org_sitk_img.GetDirection())
                direction_size = int(math.sqrt(len(org_flat_direction)))
                original_direction = org_flat_direction.reshape(direction_size, direction_size)
                if remove_index is not None and direction_size > 3:
                    original_direction = np.delete(original_direction, remove_index, 0)
                    original_direction = np.delete(original_direction, remove_index, 1)
                if slice_index is not None  and direction_size > 3:
                    original_direction = np.delete(original_direction, slice_index, 0)
                    original_direction = np.delete(original_direction, slice_index, 1)

                # submatrix_direction = original_direction[:3, :3].flatten()
                sample_sitk.SetDirection(original_direction.flatten())

            if origin is not None:
                sample_sitk.SetOrigin(tuple(origin))
            else:
                original_origin = list(org_sitk_img.GetOrigin())
                org_size = len(original_origin)
                if remove_index is not None and org_size > 3:
                    del original_origin[remove_index]
                if slice_index is not None and org_size > 3:
                    del original_origin[slice_index]

                # submatrix_direction = original_direction[:3, :3].flatten()
                sample_sitk.SetOrigin(original_origin)
        else:
            raise ValueError()
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

    @staticmethod
    def get_largest_size(*files, mood: str = "monai"):
        """
        Get the largest size for a group of datasets.
        :param files:
        :param mood: Mood that datasets are in that format. Supported format is monai.
        :return:
        """
        if mood == "monai":
            largest_size = list(SITKUtils.get_array(files[0][0]['image']).shape)
            for batch_file in files:
                for file in batch_file:
                    img = SITKUtils.get_array(file['image'])
                    for index, (l_s, i_s) in enumerate(zip(largest_size, img.shape)):
                        if i_s > l_s:
                            largest_size[index] = i_s
        else:
            raise ValueError(f"input mood: {mood} is not supported!")
        return largest_size


if __name__ == '__main__':
    array = np.array([[1, 3, 1, 2, 1], [1, 3, 2, 2, 1]])
    output = SITKUtils.swap_seg_value(array, swap_seg={1: 3, 3: 1})
    print(output)
