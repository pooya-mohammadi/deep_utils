from deep_utils.utils.lib_utils.lib_decorators import get_from_config
from deep_utils.utils.resize_utils.main_resize import resize

from .cv2_config import CV2Config


class CV2Main:
    def __init__(self, config: CV2Config):
        self.model = None
        self.config = config

    @get_from_config
    def forward(
        self,
        image,
        resize_size=None,
        img_scale_factor=None,
        img_mean=None,
        swap_rgb=None,
        resize_mode=None,
    ):
        import cv2

        image_len = len(image.shape)
        resize_img = resize(image, resize_size, mode=resize_mode)
        if image_len == 3:
            blobs = cv2.dnn.blobFromImage(
                resize_img,
                scalefactor=img_scale_factor,
                size=resize_size,
                mean=img_mean,
                swapRB=swap_rgb,
            )
        elif image_len == 4:
            blobs = cv2.dnn.blobFromImages(
                resize_img,
                scalefactor=img_scale_factor,
                size=resize_size,
                mean=img_mean,
                swapRB=swap_rgb,
            )
        else:
            raise Exception("The shape of the input image is not valid.")
        self.model.setInput(blobs)
        results = self.model.forward()
        return results
