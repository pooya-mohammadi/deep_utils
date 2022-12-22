import numpy as np

from deep_utils.utils.box_utils.boxes import Box
from deep_utils.utils.logging_utils import log_print
from deep_utils.utils.resize_utils.main_resize import resize, resize_ratio


def get_horizontal_image_stack(images, size):
    widths, heights = [], []
    for img in images:
        w, h = img.shape[:2]
        widths.append(w)
        heights.append(h)
    width = sum(widths)
    height = max(heights)
    img_base = np.zeros((width, height, 3), dtype=np.uint8)
    w_ = 0
    for img in images:
        w, h = img.shape[:2]
        img_base[w_:w_ + w, 0: h] = img
        w_ += w
    if width > size or height > size:
        img_base = resize_ratio(img_base, size)
    return img_base


def get_grid_images(
        images,
        size=(128, 128),
        n_channels=3,
        texts=None,
        text_org=None,
        text_kwargs=None,
        title=None,
        title_org=None,
        title_kwargs=None,
):
    """
    Create a group of images in a grid! Returns a numpy array.
    Args:
        images:
        size:
        n_channels:
        texts:
        text_org:
        text_kwargs:
        title:
        title_org:
        title_kwargs:

    Returns:

    """
    import math

    title_kwargs = dict() if title_kwargs is None else title_kwargs
    text_kwargs = dict() if text_kwargs is None else text_kwargs
    n_images = len(images)
    n = math.ceil(math.sqrt(len(images)))
    columns = n
    rows = math.ceil(n_images / columns)
    img_size = (rows * size[0], columns * size[1])
    img = np.zeros((img_size[0], img_size[1], n_channels), dtype=np.uint8)
    i = 0
    for r in range(rows):
        for c in range(columns):
            resized_img = resize(images[i], size)
            if texts is not None:
                org = ((size[1] // 10, size[0] // 2 - size[0] // 5)
                       if text_org is None
                       else text_org[i]
                       )
                resized_img = Box.put_text(resized_img, texts[i], org, **text_kwargs)
            if n_channels == 1 and len(resized_img.shape) == 2 and resized_img.shape[-1] !=1:
                resized_img = resized_img.reshape((*resized_img.shape[:2], 1))
            img[r * size[0]: (r + 1) * size[0], c * size[1]: (c + 1) * size[1]] = resized_img
            i += 1
            if i == len(images):
                break
        if i == len(images):
            break
    img = np.concatenate([np.zeros((size[0] // 5, img_size[1], n_channels)), img])
    if title:
        if title_org is None:
            title_org = (
                (size[1] // 10, img_size[0] // 2 - img_size[0] // 5)
                if text_org is None
                else text_org[i]
            )
        img = Box.put_text(img, text=title, org=title_org, **title_kwargs)
    img = img.astype(np.uint8)
    return img


def get_segmentation_grid_image(data_loader, save_path, n_samples=10, logger=None):
    """
    Save and get Segmentation Batches! One row is responsible for train images and the other one to mask images.
    """
    import os

    import cv2

    os.makedirs(save_path, exist_ok=True)
    c = 0
    for en, (x, y) in enumerate(data_loader):
        images = []
        for x_, y_ in zip(x, y):
            images.append(x_ * 255)
            images.append(np.stack((y_[..., 0] * 255,) * 3, axis=-1))
        img = get_grid_images(images)
        cv2.imwrite(os.path.join(
            save_path, f"batch_samples_{c}.jpg"), img[..., ::-1])
        c += 1
        if c >= n_samples:
            break
    log_print(logger, "Successfully visualized input samples!")
