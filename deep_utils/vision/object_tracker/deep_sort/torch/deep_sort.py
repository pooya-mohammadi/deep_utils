import sys
import numpy as np
import torch
from torchreid.utils import FeatureExtractor
from deep_utils.utils.color_utils.color_utils import Colors
from deep_utils.utils.box_utils.boxes import Box
from .sort.detection import Detection
from .sort.nn_matching import NearestNeighborDistanceMetric
from .sort.tracker import Tracker

sys.path.append("deep_sort/deep/reid")


class DeepSortTorchTracker:
    def __init__(self,
                 max_dist=0.2,
                 max_iou_distance=0.7,
                 max_age=70,
                 n_init=3,
                 nn_budget=100):
        max_cosine_distance = max_dist
        metric = NearestNeighborDistanceMetric(
            "euclidean", max_cosine_distance, nn_budget
        )
        self.tracker = Tracker(
            metric, max_iou_distance=max_iou_distance, max_age=max_age, n_init=n_init
        )
        self.colors = Colors()

    def visualize_output_image(self, img, update_output, confidences, names):
        if len(update_output) > 0:
            for j, (output, conf) in enumerate(zip(update_output, confidences)):
                bboxes = output[0:4]
                id = output[4]
                cls = output[5]

                c = int(cls)  # integer class
                label = f'{id} {names[c]} {conf:.2f}'
                boxes = np.array(Box.box2box(bboxes, to_source=Box.BoxSource.Numpy, in_source=Box.BoxSource.CV))
                color = self.colors(c, True)
                img = Box.put_box_text(img, boxes, label, color=color)
        return img

    def update(self, features, classes, confidences, bbox_xywh, height, width, use_yolo_preds=False):
        # self.height, self.width = height, width
        bbox_tlwh = self._xywh_to_tlwh(bbox_xywh)
        detections = [
            Detection(bbox_tlwh[i], conf, features[i])
            for i, conf in enumerate(confidences)
        ]
        # update tracker
        self.tracker.predict()
        self.tracker.update(detections, classes)

        # output bbox identities
        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            if use_yolo_preds:
                det = track.get_yolo_pred()
                x1, y1, x2, y2 = self._tlwh_to_xyxy(det.tlwh, height, width)
            else:
                box = track.to_tlwh()
                x1, y1, x2, y2 = self._tlwh_to_xyxy(box, height, width)
            track_id = track.track_id
            class_id = track.class_id
            outputs.append(
                np.array([x1, y1, x2, y2, track_id, class_id], dtype=np.int))
        if len(outputs) > 0:
            outputs = np.stack(outputs, axis=0)
        return outputs

    @staticmethod
    def _tlwh_to_xyxy(bbox_tlwh, height, width):
        """
        TODO:
            Convert bbox from xtl_ytl_w_h to xc_yc_w_h
        Thanks JieChen91@github.com for reporting this bug!
        """
        x, y, w, h = bbox_tlwh
        x1 = max(int(x), 0)
        x2 = min(int(x + w), width - 1)
        y1 = max(int(y), 0)
        y2 = min(int(y + h), height - 1)
        return x1, y1, x2, y2

    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):
        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()
        elif isinstance(bbox_xywh, torch.Tensor):
            bbox_tlwh = bbox_xywh.clone()
        bbox_tlwh[:, 0] = bbox_xywh[:, 0] - bbox_xywh[:, 2] / 2.0
        bbox_tlwh[:, 1] = bbox_xywh[:, 1] - bbox_xywh[:, 3] / 2.0
        return bbox_tlwh


class DeepSortTorchFeatureExtractor:
    def __init__(self,
                 model_name,
                 device,
                 model_path,
                 ):
        self.extractor = FeatureExtractor(
            model_name=model_name, model_path=model_path, device=str(device)
        )

    def get_features(self, bbox_xywh, ori_img):
        height, width = ori_img.shape[:2]
        im_crops = []
        for box in bbox_xywh:
            x1, y1, x2, y2 = self._xywh_to_xyxy(box, height, width)
            im = ori_img[y1:y2, x1:x2]
            im_crops.append(im)
        if im_crops:
            features = self.extractor(im_crops)
        else:
            features = np.array([])
        return features

    def _xywh_to_xyxy(self, bbox_xywh, height, width):
        x, y, w, h = bbox_xywh
        x1 = max(int(x - w / 2), 0)
        x2 = min(int(x + w / 2), width - 1)
        y1 = max(int(y - h / 2), 0)
        y2 = min(int(y + h / 2), height - 1)
        return x1, y1, x2, y2


class DeepSortTorch:
    def __init__(
            self,
            model_name,
            device,
            model_path,
            max_dist=0.2,
            max_iou_distance=0.7,
            max_age=70,
            n_init=3,
            nn_budget=100,
    ):
        self.feature_extractor = DeepSortTorchFeatureExtractor(
            model_name=model_name,
            model_path=model_path,
            device=str(device))
        self.deep_sort_tracker = DeepSortTorchTracker(
            max_dist=max_dist,
            max_iou_distance=max_iou_distance,
            max_age=max_age,
            n_init=n_init,
            nn_budget=nn_budget)

    def update(self, bbox_xywh, confidences, classes, ori_img, use_yolo_preds=False):
        height, width = ori_img.shape[:2]
        features = self.feature_extractor.get_features(bbox_xywh, ori_img)
        outputs = self.deep_sort_tracker.update(features,
                                                classes,
                                                confidences,
                                                bbox_xywh,
                                                height,
                                                width,
                                                use_yolo_preds)
        return outputs

    def visualize(self, img, tracker_output, confidences, class_names):
        img = self.deep_sort_tracker.visualize_output_image(img, tracker_output, confidences, class_names)
        return img
