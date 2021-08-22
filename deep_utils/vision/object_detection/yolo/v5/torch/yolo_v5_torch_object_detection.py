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

    def load_model(self):
        import torch
        sys.path.append(os.path.split(__file__)[0])
        from .models.experimental import attempt_load
        if self.config.model_weight:
            self.model = attempt_load(self.config.model_weight, map_location=self.config.device)
            self.model.to(self.config.device)
            self.model.eval()
            img = torch.zeros((1, 3, *self.config.img_size), device=self.config.device)
            self.model(img)
            print(f'{self.name}: weights are loaded')

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
                       get_time=False,
                       img_size=None,
                       **kwargs
                       ):
        import torch
        from .utils.datasets import letterbox
        from .utils.general import non_max_suppression, scale_coords
        im0 = img
        img = np.array([letterbox(im, new_shape=img_size)[0] for im in im0])
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

    @get_from_config
    def detect_dir(self, dir_, confidence=None,
                   iou_thresh=None, classes=None,
                   agnostic=None, img_size=None,
                   extensions=('.png', '.jpg', '.jpeg'),
                   save_result=False, res_dir=None, **kwargs):
        import cv2
        results = dict()
        for item in os.listdir(dir_):
            _, extension = os.path.splitext(item)
            if extension in extensions:
                img_path = os.path.join(dir_, item)
                img = cv2.imread(img_path)
                result = self.detect_objects(img, is_rgb=False, confidence=confidence,
                                             iou_thresh=iou_thresh, classes=classes,
                                             agnostic=agnostic, img_size=img_size,
                                             get_time=True)
                results[img_path] = result
                print(f'{img_path}: objects= {len(result["boxes"])}, time= {result["elapsed_time"]}')
                if save_result and res_dir:
                    os.makedirs(res_dir, exist_ok=True)
                    res_path = os.path.join(res_dir, item)
                    img = Box.put_box(img, result['boxes'])
                    img = Box.put_text(img, text=[f"{name}_{conf}" for name, conf in
                                                  zip(result['class_names'], result['confidences'])],
                                       org=[(b[0], b[1]) for b in result['boxes']])
                    cv2.imwrite(res_path, img)
        return results
