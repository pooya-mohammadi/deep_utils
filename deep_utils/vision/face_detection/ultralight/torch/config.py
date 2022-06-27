from deep_utils.main_abs.main_config import MainConfig


class Config(MainConfig):
    def __init__(self):
        self.device = "cpu"
        self.confidence = 0.9
        self._model_name = "slim"
        self.model = None
        self.model_urls = {
            "rbf": "https://github.com/pooya-mohammadi/deep_utils/releases/download/0.2.0/version-RFB-320.pth",
            "slim": "https://github.com/pooya-mohammadi/deep_utils/releases/download/0.2.0/version-slim-320.pth"
        }
        self.model_caches = {
            "slim": "weights/vision/face_detection/ultra-light/torch/slim/version-slim-320.pth",
            "rbf": "weights/vision/face_detection/ultra-light/torch/RBF/version-RFB-320.pth"
        }
        self.download_variables = ("model",)

    @property
    def model_url(self):
        return self.model_urls[self.model_name]

    @property
    def model_cache(self):
        return self.model_caches[self.model_name]

    @property
    def model_name(self):
        return self._model_name.lower()

    @model_name.setter
    def model_name(self, model_name):
        self._model_name = model_name
