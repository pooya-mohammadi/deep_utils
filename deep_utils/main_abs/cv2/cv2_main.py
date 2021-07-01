import cv2
from deep_utils.main_abs.main import MainClass
from deep_utils.utils.resize_utils.main_resize import resize
from .cv2_config import CV2Config


class CV2Main(MainClass):
    def __init__(self, name):
        super().__init__(name=name)
        self.config: CV2Config

    def forward(self, image):
        image_len = len(image.shape)
        resize_img = resize(self.config.resize_mode, image, self.config.resize_size)
        if image_len == 3:
            blobs = cv2.dnn.blobFromImage(
                resize_img,
                scalefactor=self.config.img_scale_factor,
                size=self.config.resize_size,
                mean=self.config.img_mean,
            )
        elif image_len == 4:
            blobs = cv2.dnn.blobFromImages(
                resize_img,
                scalefactor=self.config.img_scale_factor,
                size=self.config.resize_size,
                mean=self.config.img_mean,
            )
        else:
            raise Exception("The shape of the input image is not valid.")
        self.model.setInput(blobs)
        results = self.model.forward()
        return results
