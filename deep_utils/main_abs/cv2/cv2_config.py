from typing import Union


class CV2Config:
    def __init__(self):
        self.prototxt = None
        self.caffemodel = None
        self.resize_mode = "cv2"
        self.resize_size = (300, 300)
        self.img_mean: Union[tuple, None] = None
        self.img_scale_factor = 1
