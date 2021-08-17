import numpy as np
from deep_utils.vision.face_detection.main.main_face_detection import FaceDetector
from deep_utils.utils.lib_utils.lib_decorators import get_from_config, expand_input, get_elapsed_time, rgb2bgr
from deep_utils.utils.lib_utils.download_utils import download_decorator
from deep_utils.utils.box_utils.boxes import Box, Point
from .config import Config
import sys
from .utils.fd_config import define_img_size


define_img_size(320)
from .utils.mb_tiny_fd import create_mb_tiny_fd, create_mb_tiny_fd_predictor
from .utils.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd, create_Mb_Tiny_RFB_fd_predictor


class UltralightTorchFaceDetector(FaceDetector):
    def __init__(self, **kwargs):
        super().__init__(name=self.__class__.__name__,
                         file_path=__file__,
                         download_variables=("RBF", "slim"),
                         **kwargs)
        self.config: Config

    @download_decorator
    def load_model(self):
        # LOAD MODELS
        RBF = create_Mb_Tiny_RFB_fd(is_test=True, device=self.config.device)
        slim = create_mb_tiny_fd(is_test=True, device=self.config.device)
        return dict(RBF=RBF, slim=slim)

    @get_from_config
    @get_elapsed_time
    @rgb2bgr('rgb')
    @expand_input(3)
    def detect_faces(self,
                     images,
                     is_rgb,
                     net_type,
                     resize_size=None,
                     img_scale_factor=None,
                     img_mean=None,
                     resize_mode=None,
                     confidence=None,
                     get_time=False):
        net_type = str.lower(net_type)
        if net_type == 'slim':
            model_path = self.config.slim
            net = self.load_model()['slim']
            predictor = create_mb_tiny_fd_predictor(net, device=self.config.device)
        elif net_type == 'rbf':
            model_path = self.config.RBF
            net = self.load_model()['RBF']
            predictor = create_Mb_Tiny_RFB_fd_predictor(net, device=self.config.device)
        else:
            print("The net type is wrong!")
            sys.exit(1)

        net.load(model_path)
        if images.ndim < 4:
            images = np.array([images])

        boxes_, confidences_, landmarks_ = [], [], []
        for img_n in range(images.shape[0]):
            bboxes, labels, probs = predictor.predict(images[img_n], 500, confidence)
            conf, box = [], []
            for i in range(bboxes.size(0)):
                box_tensor = bboxes[i, :]
                boxes = [box_tensor[0].item(), box_tensor[1].item(), box_tensor[2].item(), box_tensor[3].item()]
                boxes = Box.box2box(boxes, in_source=Box.BoxSource.Torch, to_source=Box.BoxSource.Numpy)
                confidence_ = probs[i]
                if confidence_ >= confidence:
                    conf.append(confidence_.item())
                    box.append(boxes)
            confidences_.append(conf)
            boxes_.append(box)
        return dict(boxes=boxes_, confidences=confidences_, landmarks=landmarks_)
