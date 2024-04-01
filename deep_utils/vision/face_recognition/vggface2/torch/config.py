from enum import Enum

import albumentations as A
from albumentations.pytorch import ToTensorV2

from deep_utils.main_abs.main_config import MainConfig


class ModelName(str, Enum):
    SENet50 = "senet50"
    InceptionResnetV1 = "inception_resnet_v1"


class Config(MainConfig):
    def __init__(self):
        self.normalizer = "l2_normalizer"
        self.img_w = 160
        self.img_h = 160
        self.device = "cpu"
        self.model_name = "inception_resnet_v1"
        self.model = None
        self.model_urls = {
            "senet50": "https://github.com/pooya-mohammadi/deep_utils/releases/download/0.9.5/senet50_ft_weight.pkl",
            "inception_resnet_v1": "https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180402-114759-vggface2.pt"
        }
        self.model_caches = {
            "senet50": "weights/vision/face_recognition/vggface2/torch/senet50/senet50_ft_weight.pkl",
            "inception_resnet_v1": "weights/vision/face_recognition/vggface2/torch/inception_resnet_v1/20180402-114759-vggface2.pt"
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
        if self.model_name == ModelName.SENet50:
            return A.Compose(
                [A.Resize(self.img_w, self.img_h, always_apply=True),
                 A.Normalize(self.model_mean, std=self.model_std, max_pixel_value=255.0, always_apply=True),
                 ToTensorV2()
                 ], p=1)
        elif self.model_name == ModelName.InceptionResnetV1:
            return A.Compose(
                [A.Resize(self.img_w, self.img_h, always_apply=True),
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
