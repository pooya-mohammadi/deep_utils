import numpy as np
from deep_utils.utils.resize_utils.main_resize import resize
from deep_utils.utils.box_utils.boxes import Box


def group_show(images, size=(128, 128), n_channels=3, texts=None, text_org=None, text_kwargs=None, title=None,
               title_org=None, title_kwargs=None):
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
                org = (size[1] // 10, size[0] // 2 - size[0] // 5) if text_org is None else text_org[i]
                resized_img = Box.put_text(resized_img, texts[i], org, **text_kwargs)
            img[r * size[0]:(r + 1) * size[0], c * size[1]:(c + 1) * size[1]] = resized_img
            i += 1
            if i == len(images):
                break
        if i == len(images):
            break
    img = np.concatenate([np.zeros((size[0] // 5, img_size[1], 3)), img])
    if title:
        if title_org is None:
            title_org = (size[1] // 10, img_size[0] // 2 - img_size[0] // 5) if text_org is None else text_org[i]
        img = Box.put_text(img, text=title, org=title_org, **title_kwargs)
    img = img.astype(np.uint8)
    return img
