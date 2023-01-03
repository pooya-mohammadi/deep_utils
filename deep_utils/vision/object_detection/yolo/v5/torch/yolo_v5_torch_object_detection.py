import sys
import time
from pathlib import Path
from typing import Dict, List, Type, Union

import numpy as np
import torch

from deep_utils.utils.box_utils.boxes import Box
from deep_utils.utils.dict_named_tuple_utils import dictnamedtuple
from deep_utils.utils.lib_utils.lib_decorators import (
    in_shape_fix,
    lib_rgb2bgr,
    out_shape_fix,
)
from deep_utils.vision.object_detection.yolo.yolo_detector import YOLOObjectDetector
from .config import Config

OUTPUT_CLASS = dictnamedtuple(
    "Object", ["class_indices", "boxes", "confidences", "class_names", "elapsed_time"]
)


class OutputType:
    class_indices: List[int]
    boxes: List[List[int]]
    confidences: List[float]
    class_names: List[str]
    elapsed_time: float


class YOLOV5TorchObjectDetector(YOLOObjectDetector):
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
        super(YOLOV5TorchObjectDetector, self).__init__(
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

    @staticmethod
    def yolo_resize(
            img,
            new_shape=(640, 640),
            color=(114, 114, 114),
            auto=True,
            scaleFill=False,
            scaleup=True,
    ):
        from .yolov5_utils.datasets import letterbox
        return letterbox(
            img,
            new_shape=new_shape,
            color=color,
            auto=auto,
            scaleFill=scaleFill,
            scaleup=scaleup,
        )

    def load_model(self):
        FILE = Path(__file__).resolve()
        ROOT = str(FILE.parents[0])  # YOLOv5 root directory
        if ROOT not in sys.path:
            sys.path.append(ROOT)  # add ROOT to PATH
            v7_root = ROOT.replace("v5", "v7")
            if v7_root in sys.path:
                sys.path.remove(v7_root)
        from .models.experimental import attempt_load
        self.model = attempt_load(
            self.config.model_weight, device=self.config.device
        )
        self.model.to(self.config.device)
        self.model.eval()
        img = torch.zeros((1, 3, *self.config.img_size), device=self.config.device)
        self.model(img)
        print(f"{self.name}: weights are loaded")

    def detect_objects(
            self,
            img,
            is_rgb,
            class_indices=None,
            confidence=None,
            iou_thresh=None,
            img_size=None,
            agnostic=None,
            get_time=False,
            logger=None,
            verbose=1
    ) -> Union[Type[OutputType], Dict[str, list]]:
        """

        :param img:
        :param is_rgb: Is used with rgb2bgr. The required conversion is done automatically.
        :param confidence:
        :param iou_thresh:
        :param class_indices: target class indices, the rest will be ignored!
        :param agnostic:
        :param get_time:
        :param img_size:
        :param logger:
        :param verbose:
        :return:
        """
        from .yolov5_utils.general import non_max_suppression, scale_boxes
        tic = time.time() if get_time else 0

        self.update_config(
            confidence=confidence, iou_thresh=iou_thresh, img_size=img_size
        )
        img = lib_rgb2bgr(img, target_type="rgb", is_rgb=is_rgb)
        img = in_shape_fix(img, size=4)

        im0 = img
        img = np.array(
            [self.yolo_resize(im, new_shape=self.config.img_size)[0] for im in im0]
        )
        img = img.transpose((0, 3, 1, 2))
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.config.device)
        img = img / 255.0
        with torch.no_grad():
            prediction = self.model(img, augment=False)[0]
        prediction = non_max_suppression(
            prediction,
            self.config.confidence,
            self.config.iou_thresh,
            classes=class_indices,
            agnostic=agnostic,
        )
        boxes, class_names, classes, confidences = [
            [[] for _ in range(im0.shape[0])] for _ in range(4)
        ]
        for i, det in enumerate(prediction):  # detections per image
            if len(det):
                det[:, :4] = scale_boxes(
                    img.shape[2:], det[:, :4], im0[i].shape
                ).round()
                for *xyxy, conf, cls in reversed(det):
                    bbox = Box.box2box(
                        xyxy,
                        in_source=Box.BoxSource.Torch,
                        to_source=Box.BoxSource.Numpy,
                        return_int=True,
                    )
                    boxes[i].append(bbox)
                    confidences[i].append(round(conf.item(), 2))
                    cls = int(cls.item())
                    classes[i].append(cls)
                    class_names[i].append(self.config.class_names[cls])

        if get_time:
            toc = time.time()
            elapsed_time = toc - tic
        else:
            elapsed_time = 0

        output = OUTPUT_CLASS(
            class_indices=classes,
            boxes=boxes,
            confidences=confidences,
            class_names=class_names,
            elapsed_time=elapsed_time,
        )
        output = out_shape_fix(output)
        return output
