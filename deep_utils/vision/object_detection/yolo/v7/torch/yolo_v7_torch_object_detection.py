import os
import shutil
import sys
import time
from os.path import join, isfile
from pathlib import Path
from typing import Dict, List, Type, Union

import numpy as np
import torch
from tqdm import tqdm

from deep_utils.vision.object_detection.yolo.yolo_detector import YOLOObjectDetector
from deep_utils.utils.box_utils.boxes import Box
from deep_utils.utils.dir_utils.dir_utils import (
    dir_train_test_split,
    remove_create,
    transfer_directory_items,
    file_incremental,
)
from deep_utils.utils.lib_utils.lib_decorators import (
    get_from_config,
    in_shape_fix,
    lib_rgb2bgr,
    out_shape_fix,
)
from deep_utils.utils.logging_utils import log_print
from deep_utils.utils.opencv_utils.main import show_destroy_cv2
from deep_utils.utils.os_utils.os_path import split_extension
from deep_utils.utils.dict_named_tuple_utils import dictnamedtuple
from deep_utils.utils.shutil_utils.shutil_utils import mv_or_copy

from .config import Config


# FILE = Path(__file__).resolve()
# ROOT = FILE.parents[0]  # YOLOv5 root directory
# if str(ROOT) not in sys.path:
#     sys.path.append(str(ROOT))  # add ROOT to PATH
# from .models.experimental import attempt_load
# from .utils_.datasets import letterbox
# from .utils_.general import non_max_suppression, scale_coords
#
# OUTPUT_CLASS = dictnamedtuple(
#     "Object", ["class_indices", "boxes", "confidences", "class_names", "elapsed_time"]
# )


class OutputType:
    class_indices: List[int]
    boxes: List[List[int]]
    confidences: List[float]
    class_names: List[str]
    elapsed_time: float


class YOLOV7TorchObjectDetector(YOLOObjectDetector):
    def __init__(
            self,
            class_names=None,
            model_weight=None,
            device="cpu",
            img_size=(320, 320),
            confidence=0.4,
            iou_thresh=0.45,
            **kwargs,
    ):
        super(YOLOV7TorchObjectDetector, self).__init__(
            name=self.__class__.__name__,
            file_path=__file__,
            class_names=class_names,
            model_weight=model_weight,
            device=device,
            img_size=img_size,
            confidence=confidence,
            iou_thresh=iou_thresh,
            **kwargs,
        )
        self.config: Config

    def load_model(self):
        pass
