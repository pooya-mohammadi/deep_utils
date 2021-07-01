import cv2
from deep_utils.main_abs.cv2.cv2_main import CV2Main
from deep_utils.utils.lib_utils.download_utils import download_decorator
from .cv2_config import CV2Config


class CV2Caffe(CV2Main):
    def __init__(self, name, file_path, **kwargs):
        super().__init__(name=name)
        self.download_variables = ('prototxt', 'caffemodel')
        self.config: CV2Config
        self.load_config(file_path, **kwargs)
        self.load_model()

    @download_decorator
    def load_model(self):
        self.model = cv2.dnn.readNetFromCaffe(
            self.config.prototxt,
            self.config.caffemodel
        )
