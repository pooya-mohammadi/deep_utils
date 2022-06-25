import numpy as np

from deep_utils.utils.box_utils.boxes import Point


def resize(img, dsize, in_source="Numpy", mode="cv2", interpolation=None):
    mode = "cv2" if mode is None else mode
    if mode == "cv2":
        dsize = Point.point2point(
            dsize, in_source=in_source, to_source=Point.PointSource.CV
        )
        new_img = cv2_resize(img, dsize=dsize, interpolation=interpolation)
    elif mode.lower() == "pil":
        from PIL import Image

        img = img if isinstance(img, Image.Image) else Image.fromarray(img)
        new_img = img.resize(dsize, interpolation)
    else:
        raise ValueError(f"{mode} is not supported, supported types: cv2, ")
    return new_img


def cv2_resize(img, dsize, dst=None, fx=None, fy=None, interpolation=None):
    import cv2

    if len(img.shape) == 3:
        return cv2.resize(img, dsize, dst, fx, fy, interpolation)
    elif len(img.shape) == 4:
        return np.array(
            [cv2.resize(im, dsize, dst, fx, fy, interpolation) for im in img]
        )


def get_img_shape(img):
    if len(img.shape) == 4:
        return img.shape
    elif len(img.shape) == 3:
        return np.expand_dims(img, axis=0).shape
    else:
        raise Exception(f"shape: {img.shape} is not an image")


def resize_ratio(
    img,
    size: int,
    pad: bool = False,
    mode: str = "cv2",
    pad_val: int = 0,
    return_pad: bool = False,
):
    """
    Resize an image while keeping aspect-ratio
    Args:
        img: input image
        size: max-size
        mode: cv2 or pil
        pad: Pad the smaller size to become equal to the bigger one.
        pad_val: Value for padding
        return_pad: returns pad values for further processes, default is false. return: image, (h_top, h_bottom, w_left, w_right)

    Returns: resized image

    """
    if mode == "cv2":
        import math

        import cv2

        h, w = img.shape[:2]
        ratio = h / w
        if ratio > 1:
            h = size
            w = int(size / ratio)
            h_top, h_bottom = 0, 0
            w_pad = size - w
            w_left, w_right = math.ceil(w_pad / 2), w_pad // 2
        elif ratio < 1:
            h = int(size * ratio)
            w = size
            w_left, w_right = 0, 0
            h_pad = size - h
            h_top, h_bottom = math.ceil(h_pad / 2), h_pad // 2
        else:
            h, w = size, size
            w_left, w_right, h_bottom, h_top = 0, 0, 0, 0
        image = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)
        if pad:
            image = cv2.copyMakeBorder(
                image,
                h_top,
                h_bottom,
                w_left,
                w_right,
                cv2.BORDER_CONSTANT,
                value=pad_val,
            )
            if return_pad:
                return image, (h_top, h_bottom, w_left, w_right)
        return image
    else:
        raise ValueError(f"Requested mode: {mode} is not supported!")


def resize_save_aspect_ratio(img, size):
    import torch.nn.functional as F
    from PIL import Image
    from torchvision import transforms

    w, h = img.size
    ratio = h / w
    if ratio > 1:
        h = size
        w = int(size / ratio)
    elif ratio < 1:
        h = int(size * ratio)
        w = size
    else:
        return img.resize((size, size), Image.NEAREST)

    image = img.resize((w, h), Image.NEAREST)
    im = transforms.ToTensor()(image)
    val = h - w
    if val < 0:
        a = 0
        b = int(-val / 2)
    else:
        a = int(val / 2)
        b = 0

    result = F.pad(input=im, pad=(a, a, b, b))
    image = transforms.ToPILImage()(result)

    return image


if __name__ == "__main__":
    import cv2

    img = cv2.imread(
        "/home/ai/projects/Irancel-Service/hard_samples/0013903470.jpg")
    print(f"[INFO] input shape: {img.shape}")
    img = resize_ratio(img, 900)
    print(f"[INFO] output shape: {img.shape}")
    cv2.imshow("", img)
    cv2.waitKey(0)
