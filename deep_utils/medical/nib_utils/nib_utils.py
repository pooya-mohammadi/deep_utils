import math
from typing import Union, Optional, Sequence, Tuple, List

import numpy as np
import nibabel as nib
from nibabel.filebasedimages import FileBasedImage
from deep_utils.medical.main_utils import MainMedUtils


class NIBUtils(MainMedUtils):

    @staticmethod
    def get_largets_box(array: np.ndarray, get_info: bool = False):
        if get_info:
            info = MainMedUtils.get_largets_box(array, get_info)
            info['class'] = "nib"
            return info
        else:
            return MainMedUtils.get_largets_box(array, get_info)

    @staticmethod
    def get_largest_box_and_crop(array: np.ndarray, expand: int = 0, get_info: bool = False):
        if get_info:
            arr, info = MainMedUtils.get_largest_box_and_crop(array, expand, get_info)
            info['class'] = "nib"
            info['expand'] = expand
            return arr, info
        else:
            return MainMedUtils.get_largest_box_and_crop(array, expand, get_info)
    @staticmethod
    def get_img(filepath: str) -> FileBasedImage:
        img = nib.load(filepath)
        return img

    @staticmethod
    def get_array(filepath: str) -> np.ndarray:
        img = NIBUtils.get_img(filepath)
        array = img.get_fdata()
        return array

    @staticmethod
    def get_array_img(filepath: str) -> Union[np.ndarray, FileBasedImage]:
        img = NIBUtils.get_img(filepath)
        array = img.get_fdata()
        return array, img

    @staticmethod
    def save_sample(filepath: str, input_array: np.ndarray, *, affine=None, header=None, img: nib.Nifti1Image = None):
        if img:
            affine = img.affine
            header = img.header
        clipped_img = nib.Nifti1Image(sitk_img, affine, header)
        nib.save(clipped_img, filepath)
    @staticmethod
    def update_origin_of_cropped_image(origin, spacing, min_coordinates):
        z, x, y = min_coordinates
        min_coordinates = [x, y, z]


    @staticmethod
    def save(img, filepath: str):
        nib.save(img, filepath)

    @staticmethod
    def resize_nifti(input_path: str, output_path: Optional[str],
                     target_shape: Union[Tuple[int, ...], List[int]],
                     keep_values_in_reshape: bool = True):
        """
        Resize nifti file
        :param input_path:
        :param output_path:
        :param target_shape:
        :param keep_values_in_reshape:
        :return:
        """
        import cv2

        def resize_batch_images(images, target_shape):
            resized_images = np.zeros(target_shape + (images.shape[2],))

            for i in range(images.shape[2]):
                interpolation_flag = cv2.INTER_NEAREST if keep_values_in_reshape else cv2.INTER_LINEAR
                resized_images[..., i] = cv2.resize(images[..., i], target_shape, interpolation=interpolation_flag)

            return resized_images

        nii_data, nii_img = NIBUtils.get_array_img(input_path)
        resized_data = resize_batch_images(nii_data, target_shape[:2])

        repeat_factor = math.ceil(target_shape[-1] / resized_data.shape[2])  # make sure it has 155 slices!
        resized_data = np.tile(resized_data, repeat_factor)[:, :, :target_shape[-1]]
        resized_data = resized_data.astype(int).astype(float)
        new_img = nib.Nifti1Image(resized_data, nii_img.affine)
        if output_path:
            nib.save(new_img, output_path)
        return resized_data, new_img
