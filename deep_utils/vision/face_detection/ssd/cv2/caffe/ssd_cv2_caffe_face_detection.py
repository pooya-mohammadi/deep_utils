import numpy as np
from deep_utils.main_abs.cv2.cv2_caffe import CV2Caffe
from deep_utils.vision.face_detection.main import FaceDetector
from deep_utils.utils.lib_utils.lib_decorators import get_from_config, expand_input, get_elapsed_time
from deep_utils.utils.resize_utils.main_resize import get_img_shape


class SSDCV2CaffeFaceDetector(FaceDetector):

    def __init__(self, **kwargs):
        super().__init__(name=self.__class__.__name__, file_path=__file__, **kwargs)
        self.base = CV2Caffe(self.config)

    @get_elapsed_time
    @expand_input(3)
    @get_from_config
    def detect_faces(self,
                     img,
                     return_landmark=False,
                     resize_size=None,
                     img_scale_factor=None,
                     img_mean=None,
                     resize_mode=None,
                     swap_rgb=None,
                     confidence=None,
                     get_time=False):

        faces = self.base.forward(img,
                                  resize_size=resize_size,
                                  img_scale_factor=img_scale_factor,
                                  img_mean=img_mean,
                                  resize_mode=resize_mode,
                                  swap_rgb=swap_rgb)
        n_image, h, w, _ = get_img_shape(img)
        boxes = [[] for _ in range(n_image)]
        confidences = [[] for _ in range(n_image)]
        for i in range(faces.shape[2]):
            img_number = int(faces[0, 0, i, 0])
            face_confidence = faces[0, 0, i, 2]
            if face_confidence < confidence:
                continue
            box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
            start_x, start_y, end_x, end_y = box.astype("int")
            boxes[img_number].append([start_x, start_y, end_x, end_y])
            confidences[img_number].append(face_confidence)
        self.boxes = boxes
        self.confidences = confidences
        if return_landmark:
            return self.boxes, self.confidences, self.landmarks
        return self.boxes, self.confidences


if __name__ == '__main__':
    model = SSDCV2CaffeFaceDetector()
