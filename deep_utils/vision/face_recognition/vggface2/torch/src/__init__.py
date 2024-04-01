from torch import nn
from .senet_model import SENet
from .inception_resnet_v1 import InceptionResnetV1
MODELS = {
    "senet50": SENet,
    "inception_resnet_v1": InceptionResnetV1
}


def load_model(model_name, **kwargs) -> nn.Module:
    return MODELS[model_name](**kwargs)
