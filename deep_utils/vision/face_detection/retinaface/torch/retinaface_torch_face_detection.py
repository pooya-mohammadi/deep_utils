import sys

import numpy as np

from deep_utils.utils.box_utils.boxes import Box, Point
from deep_utils.utils.lib_utils.download_utils import download_decorator
from deep_utils.utils.lib_utils.lib_decorators import (
    expand_input,
    get_elapsed_time,
    get_from_config,
    rgb2bgr,
)
from deep_utils.vision.face_detection.main.main_face_detection import FaceDetector

from .config import Config


class RetinaFaceTorchFaceDetector(FaceDetector):
    def __init__(self, **kwargs):
        import torch

        from .src.layers.functions.prior_box import PriorBox
        from .src.load_model import load_model
        from .src.retinaface import RetinaFace
        from .src.utils.box_utils import decode, decode_landm
        from .src.utils.nms.py_cpu_nms import py_cpu_nms

        self.RetinaFace = RetinaFace
        self.load_model_weights = load_model
        self.torch = torch
        self.PriorBox = PriorBox
        self.py_cpu_nms = py_cpu_nms
        self.decode = decode
        self.decode_landm = decode_landm
        self.resize = 1
        self.pretrained_mobilenet = None
        network = Config.network.lower()

        if network == "resnet50":
            download_variables = ("resnet50",)
        elif network == "mobilenet":
            download_variables = ("mobilenet", "mobilenetV1X")
        else:
            raise ValueError(
                "The model is not supported\nsupported models: resnet50, mobilenet"
            )
        super().__init__(
            name=self.__class__.__name__,
            file_path=__file__,
            download_variables=download_variables,
            **kwargs
        )
        self.config: Config

    @download_decorator
    def load_model(self):
        cfg_resnet = Config.cfg_re50
        cfg_mobilenet = Config.cfg_mnet
        device = self.config.device
        network = str.lower(self.config.network)
        load_to_cpu = False if device == "cuda" else True
        if network == "resnet50":
            cfg, model_path = (cfg_resnet, self.config.resnet50)
            net = self.RetinaFace(cfg=cfg, phase="test")
            net = self.load_model_weights(
                net, model_path, load_to_cpu=load_to_cpu)
        elif network == "mobilenet":
            cfg, model_path = cfg_mobilenet, self.config.mobilenet
            net = self.RetinaFace(
                cfg=cfg, phase="test", pretrained_model=self.config.mobilenetV1X
            )
            net = self.load_model_weights(
                net, model_path, load_to_cpu=load_to_cpu)
        else:
            print("The network backbone is wrong!")
            sys.exit(1)
        self.config.cfg = cfg
        net.eval()
        net.to(device)
        self.config.priorbox = self.PriorBox(self.config.cfg)
        self.model = net

    @get_from_config
    @get_elapsed_time
    @rgb2bgr("bgr")
    @expand_input(3)
    def detect_faces(
        self,
        images,
        is_rgb,
        resize_size=None,
        img_scale_factor=None,
        img_mean=None,
        resize_mode=None,
        confidence=None,
        get_time=False,
        round_prec=4,
    ):
        device = self.config.device
        face_points = ["left_eye", "right_eye",
                       "nose", "mouth_left", "mouth_right"]
        images = np.float32(images)
        n_images = images.shape[0]
        images -= (104, 117, 123)
        images = images.transpose((0, 3, 1, 2))
        img = self.torch.FloatTensor(images).to(device)
        _, im_height, im_width = img[0].shape
        scale = self.torch.Tensor(
            [img[0].shape[2], img[0].shape[1], img[0].shape[2], img[0].shape[1]]
        )
        img = img.to(device)
        scale = scale.to(device)
        loc, conf, landms = self.model(img)

        priors = self.config.priorbox.forward(image_size=(im_height, im_width))
        priors = priors.to(device)
        prior_data = priors.data
        bboxes_, landmk_, scores_ = [], [], []
        for n_img in range(n_images):
            loc_data = loc.data[n_img]
            boxes = self.decode(loc_data, prior_data,
                                self.config.cfg["variance"])
            boxes = boxes * scale / self.resize
            boxes = boxes.cpu().numpy()
            scores = conf[n_img].data.cpu().numpy()[:, 1]

            landms = self.decode_landm(
                landms.data[n_img], prior_data, self.config.cfg["variance"]
            )
            scale1 = self.torch.Tensor(
                [
                    img.shape[3],
                    img.shape[2],
                    img.shape[3],
                    img.shape[2],
                    img.shape[3],
                    img.shape[2],
                    img.shape[3],
                    img.shape[2],
                    img.shape[3],
                    img.shape[2],
                ]
            )
            scale1 = scale1.to(device)
            landms = landms * scale1 / self.resize
            landms = landms.cpu().numpy()

            # ignore low scores
            inds = np.where(scores > confidence)[0]
            boxes = boxes[inds]
            landms = landms[inds]
            scores = scores[inds]

            # keep top-K before NMS
            order = scores.argsort()[::-1][: self.config.top_k]
            boxes = boxes[order]
            landms = landms[order]
            scores = scores[order]

            # do NMS
            bboxes = np.hstack((boxes, scores[:, np.newaxis])).astype(
                np.float32, copy=False
            )
            keep = self.py_cpu_nms(bboxes, self.config.nms_thresholds)
            bboxes = bboxes[keep, :]
            landms = landms[keep]

            # keep top-K faster NMS
            bboxes = bboxes[: self.config.keep_top_k, :]
            landms = landms[: self.config.keep_top_k, :]
            landmk_.append(landms)
            bboxes_.append(bboxes)
            scores_.append(scores)
        bboxes, landms, scores = np.array(
            bboxes_), np.array(landmk_), np.array(scores_)
        boxes_, confidences_, landmarks_ = [], [], []
        for num in range(len(bboxes)):
            box_tensor = bboxes[num, :]
            boxes = [[b[0], b[1], b[2], b[3]] for b in box_tensor]
            boxes = Box.box2box(
                boxes, in_source=Box.BoxSource.Torch, to_source=Box.BoxSource.Numpy
            )
            boxes_.append(boxes)
            confidences_.append(scores.round(round_prec))
            if len(landms) != 0:
                landmarks = [
                    [
                        Point.point2point(
                            (landms[num][j][2 * i], landms[num][j][2 * i + 1]),
                            in_source="Torch",
                            to_source="Numpy",
                        )
                        for i in range(5)
                    ]
                    for j in range(len(landms[num]))
                ]
            img_landmarks = []
            for i in range(len(landmarks)):
                face_dict = {}
                for points, face in zip(landmarks[i], face_points):
                    face_dict[face] = points
                img_landmarks.append(face_dict)
            landmarks_.append(img_landmarks)

        output = self.output_class(
            boxes=boxes_, confidences=confidences_, landmarks=landmarks_[00]
        )
        return output
