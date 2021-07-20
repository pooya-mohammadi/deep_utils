from typing import Union


class CV2Config:
    prototxt = None
    caffemodel = None
    resize_mode = 'cv2'
    resize_size = (300, 300)
    img_mean: Union[tuple, None] = None
    img_scale_factor = 1
