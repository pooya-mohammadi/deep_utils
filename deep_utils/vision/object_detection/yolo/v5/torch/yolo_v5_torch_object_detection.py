import os
import sys
import numpy as np
from deep_utils.utils.lib_utils.lib_decorators import get_from_config, expand_input, get_elapsed_time, rgb2bgr
from deep_utils.vision.object_detection.main.main_object_detection import ObjectDetector
from deep_utils.utils.box_utils.boxes import Box
from .config import Config


class YOLOV5TorchObjectDetector(ObjectDetector):
    def __init__(self, **kwargs):
        super(YOLOV5TorchObjectDetector, self).__init__(name=self.__class__.__name__, file_path=__file__, **kwargs)
        self.config: Config

    @rgb2bgr('rgb')
    @get_elapsed_time
    @expand_input(3)
    @get_from_config
    def detect_objects(self,
                       img,
                       is_rgb,
                       confidence=None,
                       iou_thresh=None,
                       classes=None,
                       agnostic=None,
                       get_time=False
                       ):
        import torch
        from .utils.datasets import letterbox
        from .utils.general import non_max_suppression, scale_coords
        im0 = img
        img = np.array([letterbox(im, new_shape=self.config.img_size)[0] for im in im0])
        img = img.transpose((0, 3, 1, 2))
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.config.device)
        img = img / 255.0
        with torch.no_grad():
            prediction = self.model(img, augment=False)[0]
        prediction = non_max_suppression(prediction, confidence, iou_thresh, classes=classes, agnostic=agnostic)
        self.boxes, self.class_names, self.classes, self.confidences = [[[] for _ in range(im0.shape[0])] for _ in
                                                                        range(4)]
        for i, det in enumerate(prediction):  # detections per image
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0[i].shape).round()
                for *xyxy, conf, cls in reversed(det):
                    bbox = Box.box2box(xyxy,
                                       in_source=Box.BoxSource.Torch,
                                       to_source=Box.BoxSource.Numpy,
                                       return_int=True)
                    self.boxes[i].append(bbox)
                    self.confidences[i].append(round(conf.item(), 2))
                    cls = int(cls.item())
                    self.classes[i].append(cls)
                    self.class_names[i].append(self.config.class_names[cls])
        return dict(
            classes=self.classes,
            boxes=self.boxes,
            confidences=self.confidences,
            class_names=self.class_names
        )

    def load_model(self):
        import torch
        sys.path.append(os.path.split(__file__)[0])
        from .models.experimental import attempt_load

        self.model = attempt_load(self.config.model_weight, map_location=self.config.device)
        self.model.to(self.config.device)
        self.model.eval()
        img = torch.zeros((1, 3, *self.config.img_size), device=self.config.device)
        self.model(img)
