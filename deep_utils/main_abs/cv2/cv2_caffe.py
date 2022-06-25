from deep_utils.main_abs.cv2.cv2_main import CV2Main
from deep_utils.utils.lib_utils.download_utils import download_decorator

from .cv2_config import CV2Config


class CV2Caffe(CV2Main):
    def __init__(self, config: CV2Config):
        super().__init__(config=config)
        self.download_variables = ("prototxt", "caffemodel")
        self.load_model()

    @download_decorator
    def load_model(self):
        import cv2

        self.model = cv2.dnn.readNetFromCaffe(
            self.config.prototxt, self.config.caffemodel
        )
