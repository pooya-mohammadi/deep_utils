import os
from os.path import join
import sys
import numpy as np
from deep_utils.utils.lib_utils.lib_decorators import get_from_config, expand_input, get_elapsed_time, rgb2bgr
from deep_utils.vision.object_detection.main.main_object_detection import ObjectDetector
from deep_utils.utils.box_utils.boxes import Box, Point
from deep_utils.utils.os_utils.os_path import split_extension
from deep_utils.utils.dir_utils.main import dir_train_test_split, transfer_directory_items
from deep_utils.utils.dir_utils.main import remove_create
from deep_utils.utils.opencv.main import show_destroy_cv2
from .config import Config


class YOLOV5TorchObjectDetector(ObjectDetector):
    def __init__(self, **kwargs):
        super(YOLOV5TorchObjectDetector, self).__init__(name=self.__class__.__name__, file_path=__file__, **kwargs)
        self.config: Config

    @staticmethod
    def yolo_resize(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
        from .utils.datasets import letterbox
        return letterbox(img, new_shape=new_shape, color=color, auto=auto, scaleFill=scaleFill, scaleup=scaleup)

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
        from .utils.general import non_max_suppression, scale_coords
        im0 = img
        img = np.array([self.yolo_resize(im, new_shape=img_size)[0] for im in im0])
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

    @staticmethod
    def test_label(img_path, str_path):
        import cv2
        img = cv2.imread(img_path)
        boxes, texts, orgs = [], [], []
        with open(str_path, mode='r') as f:
            for line in f.readlines():
                label, xc, yc, w, h = line.strip().split()
                xc, yc, w, h = float(xc), float(yc), float(w), float(h)
                boxes.append([xc, yc, w, h])
                texts.append(f"label: {label}")
                org = Point.point2point((xc - w / 2, yc - h / 2),
                                        in_source='CV',
                                        to_source='Numpy',
                                        in_relative=True,
                                        to_relative=False,
                                        shape_source='Numpy',
                                        shape=img.shape[:2])
                orgs.append(org)
        img = Box.put_box(img, boxes, in_source='CV', in_format=Box.BoxFormat.XCYC, in_relative=True)
        img = Box.put_text(img, texts, org=orgs, thickness=3, fontScale=3)
        show_destroy_cv2(img)
        return img

    @staticmethod
    def split_dataset(base_dir, out_dir, mode='cp', test_size=0.2, remove_out_dir=False, skip_transfer=True):
        img_train_names, img_val_names = dir_train_test_split(join(base_dir, 'images'),
                                                              train_dir=join(out_dir, 'train', 'images'),
                                                              val_dir=join(out_dir, 'val', 'images'),
                                                              mode=mode,
                                                              remove_out_dir=remove_out_dir,
                                                              test_size=test_size)
        img_train_labels = [os.path.splitext(name)[0] + '.txt' for name in img_train_names]
        img_val_labels = [os.path.splitext(name)[0] + '.txt' for name in img_val_names]

        transfer_directory_items(join(base_dir, 'labels'), join(out_dir, 'train', 'labels'),
                                 img_train_labels, mode=mode, remove_out_dir=remove_out_dir,
                                 skip_transfer=skip_transfer)
        transfer_directory_items(join(base_dir, 'labels'), join(out_dir, 'val', 'labels'), img_val_labels,
                                 mode=mode, remove_out_dir=remove_out_dir, skip_transfer=skip_transfer)

    @staticmethod
    def extract_label(label_path, img_path=None, shape=None, shape_source=None):
        with open(label_path, mode='r') as f:
            boxes, labels = [], []
            for line in f.readlines():
                label, b1, b2, b3, b4 = line.strip().split()
                boxes.append([float(b1), float(b2), float(b3), float(b4)])
                labels.append(int(label))

        if img_path is not None:
            import cv2
            shape = cv2.imread(img_path).shape[:2]
            shape_source = 'Numpy'
        if shape is not None and shape_source is not None:
            boxes = Box.box2box(boxes,
                                in_source=Box.BoxSource.CV,
                                to_source='Numpy',
                                in_format='XCYC',
                                to_format='XYXY',
                                in_relative=True,
                                to_relative=False,
                                shape=shape,
                                shape_source=shape_source)
        return boxes, labels

    @get_from_config
    def detect_dir(self,
                   dir_,
                   confidence=None,
                   iou_thresh=None,
                   classes=None,
                   agnostic=None,
                   img_size=None,
                   extensions=('.png', '.jpg', '.jpeg'),
                   res_img_dir=None,
                   res_label_dir=None,
                   put_annotations=True,
                   remove_dirs=False,
                   **kwargs):
        import cv2
        results = dict()
        if res_label_dir and remove_dirs:
            remove_create(res_label_dir)
        if res_img_dir and remove_dirs:
            remove_create(res_img_dir)
        for item_name in os.listdir(dir_):
            _, extension = os.path.splitext(item_name)
            if extension in extensions:
                img_path = os.path.join(dir_, item_name)
                img = cv2.imread(img_path)
                result = self.detect_objects(img,
                                             is_rgb=False,
                                             confidence=confidence,
                                             iou_thresh=iou_thresh,
                                             classes=classes,
                                             agnostic=agnostic,
                                             img_size=img_size,
                                             get_time=True)
                results[img_path] = result
                print(f'{img_path}: objects= {len(result["boxes"])}, time= {result["elapsed_time"]}')
                boxes = result['boxes']
                if len(result['boxes']):
                    if res_img_dir:
                        res_path = os.path.join(res_img_dir, item_name)
                        if put_annotations:
                            img = Box.put_box(img, boxes)
                            img = Box.put_text(img, text=[f"{name}_{conf}" for name, conf in
                                                          zip(result['class_names'], result['confidences'])],
                                               org=[(b[0], b[1]) for b in result['boxes']])
                        cv2.imwrite(res_path, img)
                    if res_label_dir:
                        res_path = os.path.join(res_label_dir, split_extension(item_name, '.txt'))
                        xcyc_boxes = Box.box2box(boxes,
                                                 in_format='XYXY',
                                                 to_format='XCYC',
                                                 in_source='Numpy',
                                                 to_source='CV',
                                                 in_relative=False,
                                                 to_relative=True,
                                                 shape_source='Numpy',
                                                 shape=img.shape[:2])
                        with open(res_path, mode='w') as f:
                            for (b1, b2, b3, b4), class_ in zip(xcyc_boxes, result['classes']):
                                f.write(f'{class_} {b1} {b2} {b3} {b4}\n')

        return results

    @staticmethod
    def clean_samples(label_path, img_path, ext='.jpg'):
        img_names = [os.path.splitext(img)[0] for img in os.listdir(img_path)]
        label_names = [os.path.splitext(l)[0] for l in os.listdir(label_path)]
        for label in label_names:
            if label not in img_names:
                os.remove(join(label_path, label + ".txt"))
        for img_name in img_names:
            if img_name not in label_names:
                os.remove(join(img_path, img_name + ext))
