import sys

import cv2
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
from .utils.rfb_320 import create_rfb_net
from .utils.slim_320 import create_slim_net


class UltralightTFFaceDetector(FaceDetector):
    def __init__(self, **kwargs):
        super().__init__(
            name=self.__class__.__name__,
            file_path=__file__,
            download_variables=("RBF", "slim"),
            **kwargs
        )
        self.config: Config
        import tensorflow as tf

        self.tf = tf
        physical_devices = tf.config.experimental.list_physical_devices("GPU")
        if len(physical_devices) < 0:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)

    @download_decorator
    def load_model(self):
        # LOAD MODELS
        RBF = create_rfb_net()
        slim = create_slim_net()

    @get_from_config
    @get_elapsed_time
    @rgb2bgr("rgb")
    @expand_input(3)
    def detect_faces(
        self,
        images,
        is_rgb,
        net_type,
        resize_size=None,
        img_scale_factor=None,
        img_mean=None,
        resize_mode=None,
        confidence=None,
        get_time=False,
    ):
        net_type = str.lower(net_type)
        if net_type == "slim":
            model_path = self.config.slim
        elif net_type == "rbf":
            model_path = self.config.RBF
        else:
            print("The net type is wrong!")
            sys.exit(1)

        model = self.tf.keras.models.load_model(model_path)
        if images.ndim < 4:
            images = np.array([images])

        boxes_, confidences_, landmarks_ = [], [], []
        for img_n in range(images.shape[0]):
            h, w, _ = images[img_n].shape
            img_resize = cv2.resize(images[img_n], (320, 240))
            img_resize = img_resize - 127.0
            img_resize = img_resize / 128.0
            img_resize = np.expand_dims(img_resize, axis=0)

            # result=[background,face,x1,y1,x2,y2]
            results = model.predict(img_resize)
            conf, box = [], []
            for result in results:
                start_x = int(result[2] * w)
                start_y = int(result[3] * h)
                end_x = int(result[4] * w)
                end_y = int(result[5] * h)
                boxes = [start_x, start_y, end_x, end_y]
                boxes = Box.box2box(
                    boxes, in_source=Box.BoxSource.CV, to_source=Box.BoxSource.Numpy
                )
                confidence_ = result[1]
                if confidence_ >= confidence:
                    conf.append(confidence_)
                    box.append(boxes)
            confidences_.append(conf)
            boxes_.append(box)
        return dict(boxes=boxes_, confidences=confidences_, landmarks=landmarks_)
