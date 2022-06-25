import torch.nn as nn
from torchvision import models

from deep_utils.blocks.torch.blocks_torch import BlocksTorch
from deep_utils.utils.logging_utils.logging_utils import value_error_log


class TorchVisionModel(nn.Module):
    """
    A model that loads torch-vision models with an extra convolution layer at the end for classification
    """

    def __init__(
        self,
        model_name,
        num_classes,
        last_layer_nodes,
        use_pretrained=True,
        feature_extract=True,
        logger=None,
    ):
        super(TorchVisionModel, self).__init__()
        if model_name == "squeezenet":
            self.model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        else:
            value_error_log(
                logger, f"model_name: {model_name} is not supported!")
        BlocksTorch.set_parameter_requires_grad(self, feature_extract)
        self.model_ft.classifier[1] = nn.Conv2d(
            last_layer_nodes, num_classes, kernel_size=(1, 1), stride=(1, 1)
        )

    def forward(self, inputs):
        output = self.model_ft(inputs)

        return output
