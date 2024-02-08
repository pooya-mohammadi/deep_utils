from typing import Union

import numpy as np
import nibabel as nib
from nibabel.filebasedimages import FileBasedImage


class NIBUtils:
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
    def get_img_array(filepath: str) -> Union[np.ndarray, FileBasedImage]:
        img = NIBUtils.get_img(filepath)
        array = img.get_fdata()
        return array, img

    @staticmethod
    def save_sample(filepath: str, sample_array, affine, header):
        clipped_img = nib.Nifti1Image(sample_array, affine, header)
        nib.save(clipped_img, filepath)


