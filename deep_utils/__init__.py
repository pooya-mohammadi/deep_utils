from typing import TYPE_CHECKING

from deep_utils._dummy_objects.dummy_framework.dummy_framework import _LazyModule
from deep_utils._dummy_objects.dummy_framework.dummy_framework import (
    is_torch_available,
    is_tf_available,
    is_transformers_available,
    is_cv2_available,
    is_torchvision_available,
    is_monai_available,
    is_timm_available,
)

# Deep Utils version number
__version__ = "1.0.0"

# no third-party python libraries are required for the following classes
_import_structure = {
    "utils.box_utils.boxes": ["Box", "Point"],
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
    "utils.pickle_utils.pickle_utils": ["dump_pickle", "load_pickle"],

}

if is_timm_available() and is_transformers_available() and is_torchvision_available() and is_torch_available():
    _import_structure["vision.image_caption.blip.torch.blip_torch_image_caption"] = ["BlipTorchImageCaption"]
else:
    from ._dummy_objects import torchvision_torch_timm_transformers_fairscale_dummy

    _import_structure["_dummy_objects.torchvision_torch_timm_transformers_fairscale_dummy"] = [
        name for name in dir(torchvision_torch_timm_transformers_fairscale_dummy) if not name.startswith("_")
    ]

if is_tf_available():
    _import_structure["utils.tf_utils.main"] = ["TFUtils"]
else:
    from ._dummy_objects import tf_dummy

    _import_structure["_dummy_objects.tf_dummy"] = [
        name for name in dir(tf_dummy) if not name.startswith("_")
    ]

if is_torch_available() and is_monai_available():
    _import_structure['preprocessing.monai.monai_segmentation'] = ["MonaiChannelBasedContrastEnhancementD"]
else:
    from ._dummy_objects import torch_monai_dummy

    _import_structure["_dummy_objects.torch_monai_dummy"] = [
        name for name in dir(torch_monai_dummy) if not name.startswith("_")
    ]

if is_torch_available():
    _import_structure["callbacks.torch.torch_tensorboard"] = ["TensorboardTorch"]
    _import_structure["utils.torch_utils.torch_utils"] = ["TorchUtils"]
    _import_structure["blocks.torch.blocks_torch"] = ["BlocksTorch"]
    _import_structure["vision.color_recognition.cnn_color.torch.color_cnn_torch"] = ["ColorRecognitionCNNTorch"]
else:
    from ._dummy_objects import torch_dummy

    _import_structure["_dummy_objects.torch_dummy"] = [
        name for name in dir(torch_dummy) if not name.startswith("_")
    ]

if is_cv2_available():
    _import_structure["vision.face_detection.haarcascade.cv2_.haarcascade_cv2_face_detection"] = [
        "HaarcascadeCV2FaceDetector"]
    _import_structure["utils.opencv_utils.opencv_utils"] = ["CVUtils", "show_destroy_cv2"]
    _import_structure["utils.encodes.b64"] = ["b64_to_img", "img_to_b64"]
else:
    from ._dummy_objects import cv2_dummy

    _import_structure["_dummy_objects.cv2_dummy"] = [
        name for name in dir(cv2_dummy) if not name.startswith("_")
    ]

if is_torch_available() and is_cv2_available():
    _import_structure["vision.face_detection.mtcnn.torch.mtcnn_torch_face_detection"] = ["MTCNNTorchFaceDetector"]
    _import_structure["vision.object_detection.yolo.v5.torch.yolo_v5_torch_object_detection"] = [
        "YOLOV5TorchObjectDetector"]
    _import_structure["vision.object_detection.yolo.v7.torch.yolo_v7_torch_object_detection"] = [
        "YOLOV7TorchObjectDetector"]
else:
    from ._dummy_objects import torch_cv2_dummy

    _import_structure["_dummy_objects.torch_cv2_dummy"] = [
        name for name in dir(torch_cv2_dummy) if not name.startswith("_")
    ]

if is_tf_available() and is_cv2_available():
    _import_structure["vision.face_detection.mtcnn.tf.mtcnn_tf_face_detection"] = ["MTCNNTFFaceDetector"]
else:
    from ._dummy_objects import tf_cv2_dummy

    _import_structure["_dummy_objects.tf_cv2_dummy"] = [
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
    from .utils.tf_utils.main import TFUtils
    from .vision.object_detection.yolo.v5.torch.yolo_v5_torch_object_detection import YOLOV5TorchObjectDetector
    from .vision.object_detection.yolo.v7.torch.yolo_v7_torch_object_detection import YOLOV7TorchObjectDetector
    from .utils.opencv_utils.opencv_utils import CVUtils
    from vision.image_caption.image_caption import ImageCaption
    from vision.image_caption.blip.torch.blip_torch_image_caption import BlipTorchImageCaption
    from utils.encodes.b64 import b64_to_img
    from blocks.torch.blocks_torch import BlocksTorch
    from vision.color_recognition.cnn_color.torch.color_cnn_torch import ColorRecognitionCNNTorch
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
        extra_objects={"__version__": __version__},
    )
