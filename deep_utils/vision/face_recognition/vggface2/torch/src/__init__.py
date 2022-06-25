from torch import nn
from .senet_model import SENet

MODELS = {
    "senet50": SENet
}


def load_model(model_name, **kwargs) -> nn.Module:
    return MODELS[model_name](**kwargs)
