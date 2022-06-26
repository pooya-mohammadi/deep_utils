from deep_utils.main_abs.main_config import MainConfig
import albumentations as A
from albumentations.pytorch import ToTensorV2


class Config(MainConfig):
    def __init__(self):
        self.normalizer = "l2_normalizer"
        self.img_w = 224
        self.img_h = 224
        self.device = "cpu"
        self.model_name = "senet50"
        self.model = None
        self.model_urls = {
            "senet50": "https://github.com/pooya-mohammadi/deep_utils/releases/download/0.9.5/senet50_ft_weight.pkl"
        }
        self.model_caches = {
            "senet50": "weights/vision/face_recognition/vggface2/torch/senet50/senet50_ft_weight.pkl"
        }
        self.download_variables = ("model",)

        self.model_means = {
            "senet50": [91.4953, 103.8827, 131.0912]
        }
        self.model_stds = {
            "senet50": [1, 1, 1]
        }

    @property
    def transform(self):
        return A.Compose(
            [A.Resize(self.img_w, self.img_h, always_apply=True),
             A.Normalize(self.model_mean, std=self.model_std, max_pixel_value=255.0, always_apply=True),
             ToTensorV2()
             ], p=1)

    @property
    def model_url(self):
        return self.model_urls[self.model_name]

    @property
    def model_cache(self):
        return self.model_caches[self.model_name]

    @property
    def model_mean(self):
        return self.model_means[self.model_name]

    @property
    def model_std(self):
        return self.model_stds[self.model_name]
