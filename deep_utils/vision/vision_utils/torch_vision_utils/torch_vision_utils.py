import torch
from PIL import Image


class TorchVisionUtils:
    @staticmethod
    def transform_concatenate_images(images, transform, mode=None, device='cpu'):
        """
        Concat and transform a batch images
        :param images:
        :param transform:
        :param mode:
        :param device:
        :return:
        """
        if mode == 'torchvision' or 'torchvision' in str(transform.__class__):
            output = torch.cat([transform(Image.fromarray(sample)).unsqueeze(0) for sample in images]).to(device)
        elif mode == "albumentation" or "albumentation" in str(transform.__class__):
            output = torch.cat([transform(image=sample)["image"].unsqueeze(0) for sample in images]).to(device)
        else:
            raise ValueError(f"mode: {mode} is not supported! [albumentation, torchvision]")
        return output
