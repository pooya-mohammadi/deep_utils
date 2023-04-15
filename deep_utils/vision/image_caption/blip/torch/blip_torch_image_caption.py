from pathlib import Path
from typing import Tuple, Union

import PIL.Image
import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from deep_utils.multi_modals._models.blip.torch.models.blip import blip_decoder
from deep_utils.vision.image_caption.image_caption import ImageCaption


class BlipTorchImageCaption(ImageCaption):
    def __init__(self, weight_path: str, device="cuda", img_w: int = 384, img_h: int = 384,
                 img_norm_mean: Tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073),
                 img_norm_std: Tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711),
                 vit='base'
                 ):
        super().__init__(weight_path=weight_path, device=device, img_w=img_w, img_h=img_h, img_norm_std=img_norm_std,
                         img_norm_mean=img_norm_mean)
        self._transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_w, img_h), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(img_norm_mean, img_norm_std)
        ])

        self._model = blip_decoder(pretrained=self._weight_path, image_size=self._img_w, vit=vit).to(
            self._device).eval()

    def generate_caption(self, image: Union[np.array, str, Path], sample=False, num_beams=3, max_length=20,
                         min_length=5) -> str:
        if isinstance(image, str) or isinstance(image, Path):
            image = PIL.Image.open(image)
            image = np.array(image)
        if len(image.shape) == 2:
            image = image[..., None]
            image = np.concatenate([image] * 3, axis=-1)
        image = self._transform(image)
        image = image.to(self._device)

        with torch.no_grad():
            image = image.unsqueeze(0)
            caption = self._model.generate(image, sample=sample, num_beams=num_beams, max_length=max_length,
                                           min_length=min_length)
        return caption[0]
