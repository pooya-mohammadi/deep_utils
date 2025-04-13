from pathlib import Path
from typing import Union

import cv2
import numpy as np
import torch

from deep_utils.utils.torch_utils.torch_utils import TorchUtils
from deep_utils.vision.color_recognition.cnn_color.torch.color_cnn_torch import ColorRecognitionCNNTorch


class ColorRecognitionCNNTorchPrediction:
    def __init__(self, model_path, device='cpu'):
        self.device = device
        if torch.__version__ >= "2.6.0":
            state_dict = torch.load(model_path, map_location=self.device, weights_only=False)
        else:
            state_dict = torch.load(model_path, map_location=self.device)
        self.model = ColorRecognitionCNNTorch(n_classes=state_dict['n_classes'],
                                              in_channel=state_dict.get('in_channel', 3))
        self.model.to(self.device)
        TorchUtils.load_model(self.model, state_dict['state_dict'])
        # skip cold start
        self.model(torch.randn((1, state_dict.get('in_channel', 3),
                                state_dict.get("width", 224),
                                state_dict.get("height", 224))).to(device))
        self.model.eval()

        self.id2class = state_dict['id2class']
        self.transform = state_dict['val_transform']
        del state_dict

    def detect(self, img: Union[str, Path, np.ndarray]) -> str:
        if type(img) is not np.ndarray:
            img = cv2.imread(img)[..., ::-1]
        image = self.transform(image=img)["image"]
        image = image.view(1, *image.size()).to(self.device)
        with torch.no_grad():
            logits = self.model(image).cpu().squeeze(0).numpy()
        prediction = np.argmax(logits)
        return self.id2class[prediction]
