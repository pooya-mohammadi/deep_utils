import os
from pathlib import Path
from time import time
from typing import Union

import cv2
import numpy as np
import torch

from deep_utils.utils.logging_utils.logging_utils import log_print
from deep_utils.vision.torch_vision.torch_vision_models import TorchVisionModel
from deep_utils.vision.vision_utils.torch_vision_utils.torch_vision_utils import TorchVisionUtils


class TorchVisionInference:
    def __init__(self, model_path, device="cpu", logger=None, verbose=1):
        """
        A class for model inference
        :param model_path: Path to the model weights and params
        :param device: TorchVisionInference device
        :param logger:
        :param verbose:
        """
        self.logger = logger
        self.verbose = verbose
        save_params = torch.load(model_path, map_location=device)
        self.model = TorchVisionModel(
            model_name=save_params["model_name"],
            num_classes=save_params["n_classes"],
            last_layer_nodes=save_params["last_layer_nodes"],
            use_pretrained=False,
            feature_extract=True,
        )
        try:
            self.model.load_state_dict(save_params["state_dict"])
        except:
            self.model.load_state_dict(
                {
                    ".".join(k.split(".")[1:]): v
                    for k, v in save_params["state_dict"].items()
                }
            )

        self.device = device
        self.label_map = save_params["id_to_class"]
        self.model.eval()
        self.transform = save_params["val_transform"]

    def infer(self, img: Union[str, Path, np.ndarray], return_confidence=False):
        """

        :param img:
        :param return_confidence: If True, return confidence alongside the prediction class
        :return:
        """
        if isinstance(img, Path) or isinstance(img, str):
            img = cv2.imread(img)[..., ::-1]
        image = self.transform(image=img)["image"]
        image = image.view(1, *image.size())
        with torch.no_grad():
            image = image.to(self.device)
            logits = self.model(image).squeeze(0)
        confidence, prediction = torch.max(torch.softmax(logits, dim=0), dim=0)
        confidence, prediction = confidence.item(), prediction.item()
        label = self.label_map[prediction]
        if return_confidence:
            return label, confidence
        return self.label_map[prediction]

    def infer_group(self, images: np.ndarray, return_confidence=False):
        """
        Infer a batch of images
        :param images: A batch of images
        :param return_confidence: If True, return confidence alongside the prediction class
        :return:
        """
        images_tensor = TorchVisionUtils.transform_concatenate_images(images, self.transform, device=self.device)
        with torch.no_grad():
            logits = self.model(images_tensor)
        confidence, prediction = torch.max(torch.softmax(logits, dim=1), dim=1)
        labels = [self.label_map[lbl.item()] for lbl in prediction]
        if return_confidence:
            return labels, confidence.numpy()
        else:
            return labels

    def infer_directory(self, dir_path: Union[Path, str]):
        """
        Infer a directory of images!
        :param dir_path:
        :return:
        """
        for image_name in sorted(os.listdir(dir_path)):
            image_path = os.path.join(dir_path, image_name)
            try:
                tic = time()
                prediction = self.infer(image_path)
                toc = time()
                log_print(
                    self.logger,
                    f"predicted class for {image_path} is {prediction}\ninference time: {toc - tic}",
                    verbose=self.verbose,
                )
            except BaseException as e:
                log_print(
                    self.logger,
                    f"img: {image_path} is invalid -> {e}",
                    verbose=self.verbose,
                )
