import cv2
import numpy as np

from deep_utils.utils.box_utils.boxes import Point
from deep_utils.utils.resize_utils.main_resize import resize


class VideoWriterCV:
    def __init__(
            self,
            save_path,
            width,
            height,
            fourcc="XVID",
            fps=30,
            colorful=True,
            in_source="Numpy",
    ):
        self.width, self.height = width, height

        point = Point.point2point(
            (width, height), in_source=in_source, to_source=Point.PointSource.CV
        )
        fourcc = cv2.VideoWriter_fourcc(
            *fourcc) if isinstance(fourcc, str) else fourcc
        self.vw = cv2.VideoWriter(save_path, fourcc, fps, point, colorful)

    def write(self, frame):
        if frame.shape[:2] != (self.width, self.height):
            frame = resize(frame, (self.width, self.height))
        self.vw.write(frame)


class CVUtils:
    @staticmethod
    def rotate(
            img: np.ndarray,
            rotation_degree: int,
            center_point=None,
            scale=1.0,
            dsize=None,
            bound=False,
            clockwise=True,
    ):

        h, w = img.shape[:2]
        (w, h) = dsize = (w, h) if dsize is None else dsize
        center_point = (w // 2, h // 2) if center_point is None else center_point
        # negative angle >> clockwise rotation | positive angle >> counter clockwise rotation
        rotation_degree = -rotation_degree if clockwise else rotation_degree
        m = cv2.getRotationMatrix2D(center_point, rotation_degree, scale)
        if bound:
            h, w = img.shape[:2]
            cos = abs(m[0, 0])
            sin = abs(m[0, 1])
            w_ = int((cos * w) + (sin * h))
            h_ = int((cos * h) + (sin * w))
            m[0, 2] += w_ // 2 - w // 2
            m[1, 2] += h_ // 2 - h // 2
            dsize = (w_, h_)
        rotated = cv2.warpAffine(img, m, dsize)

        return rotated

    @staticmethod
    def translate(img, tx, ty, dsize=None):
        h, w = img.shape[:2][::-1]
        dsize = (w, h) if dsize is None else dsize
        translation_matrix = np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)
        translated_image = cv2.warpAffine(
            src=img, M=translation_matrix, dsize=dsize)
        return translated_image

    @staticmethod
    def show_destroy_cv2(img, win_name="", show=True):
        if show:
            try:
                cv2.imshow(win_name, img)
                cv2.waitKey(0)
                cv2.destroyWindow(win_name)
            except Exception as e:
                cv2.destroyWindow(win_name)
                raise e


show_destroy_cv2 = CVUtils.show_destroy_cv2
