import pytest
import numpy as np
from deep_utils import TorchVisionUtils


@pytest.mark.torchvision
def test_detection():
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    from torchvision.transforms import transforms

    # albumentation test
    height, width, mean, std = 224, 220, 0, 1
    transform = A.Compose(
        [A.Resize(height, width),
         A.Normalize(mean, std, max_pixel_value=255.0),
         ToTensorV2()
         ])
    img_1 = np.random.randint(0, 255, (256, 180, 3), dtype=np.uint8)
    img_2 = np.random.randint(0, 255, (224, 289, 3), dtype=np.uint8)

    images = TorchVisionUtils.transform_concatenate_images([img_1, img_2], transform=transform)
    assert tuple(images.shape) == (
        2, 3, height, width), "TorchVisionUtils transform_concatenate_images not good for albumentation"

    # TorchVision test
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((height, width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)]
    )
    images = TorchVisionUtils.transform_concatenate_images([img_1, img_2], transform=transform)
    assert tuple(images.shape) == (
        2, 1, height, width), "TorchVisionUtils transform_concatenate_images not good for torchvision"
