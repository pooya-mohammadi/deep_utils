import sys
import numpy as np
from deep_utils.utils.box_utils.boxes import Box
from deep_utils.utils.lib_utils.download_utils import download_decorator
from deep_utils.utils.lib_utils.lib_decorators import (
    expand_input,
    get_elapsed_time,
    get_from_config,
    rgb2bgr,
)
from deep_utils.vision.face_detection.main.main_face_detection import FaceDetector

from .config import Config
from .utils.fd_config import define_img_size
from .utils.mb_tiny_fd import create_mb_tiny_fd, create_mb_tiny_fd_predictor
from .utils.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd, create_Mb_Tiny_RFB_fd_predictor

define_img_size(320)


class UltralightTorchFaceDetector(FaceDetector):
    def __init__(self, **kwargs):
        super().__init__(
            name=self.__class__.__name__,
            file_path=__file__,
            download_variables=("model",),
            **kwargs
        )
        self.config: Config

    @download_decorator
    def load_model(self):
        # LOAD MODELS
        model_path = self.config.model
        if self.config.model_name == "slim":
            net = create_mb_tiny_fd(is_test=True, device=self.config.device)
            net.load(model_path)
            predictor = create_mb_tiny_fd_predictor(net, device=self.config.device)
        elif self.config.model_name == "rbf":
            net = create_Mb_Tiny_RFB_fd(is_test=True, device=self.config.device)
            net.load(model_path)
            predictor = create_Mb_Tiny_RFB_fd_predictor(net, device=self.config.device)
        else:
            print("The net type is wrong!")
            sys.exit(1)
        self.model = predictor

    @get_from_config
    @get_elapsed_time
    @rgb2bgr("rgb")
    @expand_input(3)
    def detect_faces(
            self,
            images,
            is_rgb,
            confidence=None,
            get_time=False,
    ):

        if images.ndim < 4:
            images = np.array([images])

        boxes_, confidences_, landmarks_ = [], [], []
        for img_n in range(images.shape[0]):
            bboxes, labels, probs = self.model.predict(images[img_n], 500, self.config.confidence)
            conf, box = [], []
            for i in range(bboxes.size(0)):
                box_tensor = bboxes[i, :]
                boxes = [
                    box_tensor[0].item(),
                    box_tensor[1].item(),
                    box_tensor[2].item(),
                    box_tensor[3].item(),
                ]
                boxes = Box.box2box(
                    boxes, in_source=Box.BoxSource.Torch, to_source=Box.BoxSource.Numpy
                )
                confidence_ = probs[i]
                if confidence_ >= self.config.confidence:
                    conf.append(confidence_.item())
                    box.append(boxes)
            confidences_.append(conf)
            boxes_.append(box)
        return dict(boxes=boxes_, confidences=confidences_, landmarks=landmarks_)
