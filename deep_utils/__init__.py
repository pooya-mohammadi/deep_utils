from typing import TYPE_CHECKING

from deep_utils.dummy_objects.dummy_framework.dummy_framework import _LazyModule
from deep_utils.dummy_objects.dummy_framework.dummy_framework import (
    is_torch_available,
    is_tf_available,
    is_transformers_available,
    is_cv2_available,
    is_torchvision_available,
    is_monai_available,
)

# Deep Utils version number
__version__ = "1.0.0"

# no third-party python libraries are required for the following classes
_import_structure = {
    "utils.box_utils": ["Box", "Point"],
    "utils.os_utils.os_path": ["validate_file_extension", "is_img", "split_extension", "split_all", "get_file_name"],
    "utils.dir_utils.dir_utils": ["transfer_directory_items",
                                  "dir_train_test_split",
                                  "split_dir_of_dir",
                                  "split_xy_dir",
                                  "crawl_directory_dataset",
                                  "remove_create",
                                  "mkdir_incremental",
                                  "file_incremental",
                                  "cp_mv_all",
                                  "split_segmentation_dirs",
                                  "find_file",
                                  "combine_directory_of_directories"],
}
if is_torch_available() and is_monai_available():
    _import_structure['preprocessing.monai.monai_segmentation'] = ["MonaiChannelBasedContrastEnhancementD"]
else:
    from .dummy_objects import torch_monai_dummy

    _import_structure["dummy_objects.torch_monai_dummy"] = [
        name for name in dir(torch_monai_dummy) if not name.startswith("_")
    ]

if is_torch_available():
    _import_structure["callbacks.torch.torch_tensorboard"] = ["TensorboardTorch"]
    _import_structure["utils.torch_utils.torch_utils"] = ["TorchUtils"]
else:
    from .dummy_objects import torch_dummy

    _import_structure["dummy_objects.torch_dummy"] = [
        name for name in dir(torch_dummy) if not name.startswith("_")
    ]

if is_cv2_available():
    _import_structure["vision.face_detection.haarcascade.cv2_.haarcascade_cv2_face_detection"] = [
        "HaarcascadeCV2FaceDetector"]
else:
    from .dummy_objects import cv2_dummy

    _import_structure["dummy_objects.cv2_dummy"] = [
        name for name in dir(cv2_dummy) if not name.startswith("_")
    ]

if is_torch_available() and is_cv2_available():
    _import_structure["vision.face_detection.mtcnn.torch.mtcnn_torch_face_detection"] = ["MTCNNTorchFaceDetector"]
else:
    from .dummy_objects import torch_cv2_dummy

    _import_structure["dummy_objects.torch_cv2_dummy"] = [
        name for name in dir(torch_cv2_dummy) if not name.startswith("_")
    ]

if is_tf_available() and is_cv2_available():
    _import_structure["vision.face_detection.mtcnn.tf.mtcnn_tf_face_detection"] = ["MTCNNTFFaceDetector"]
else:
    from .dummy_objects import tf_cv2_dummy

    _import_structure["dummy_objects.tf_cv2_dummy"] = [
        name for name in dir(tf_cv2_dummy) if not name.startswith("_")
    ]

if TYPE_CHECKING:
    from .utils.box_utils.boxes import Box, Point
    from .vision.face_detection.haarcascade.cv2_.haarcascade_cv2_face_detection import HaarcascadeCV2FaceDetector
    from .vision.face_detection.mtcnn.tf.mtcnn_tf_face_detection import MTCNNTFFaceDetector
    from .vision.face_detection.mtcnn.torch.mtcnn_torch_face_detection import MTCNNTorchFaceDetector
    from .vision.face_detection.haarcascade.cv2_.haarcascade_cv2_face_detection import HaarcascadeCV2FaceDetector
    from .callbacks.torch.torch_tensorboard import TensorboardTorch
    from .utils.os_utils.os_path import validate_file_extension, is_img, split_extension, split_all, get_file_name
    from .utils.dir_utils.dir_utils import (transfer_directory_items, dir_train_test_split, split_dir_of_dir,
                                            split_xy_dir, crawl_directory_dataset, remove_create, mkdir_incremental,
                                            file_incremental, cp_mv_all, split_segmentation_dirs, find_file,
                                            combine_directory_of_directories)
    from .preprocessing.monai.monai_segmentation import MonaiChannelBasedContrastEnhancementD
    from .utils.torch_utils.torch_utils import TorchUtils
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
        extra_objects={"__version__": __version__},
    )
