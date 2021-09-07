from abc import abstractmethod
from deep_utils.main_abs.main import MainClass
from deep_utils.utils.utils.main import dictnamedtuple

OUTPUT_CLASS = dictnamedtuple("FaceDetector", ["boxes", "landmarks", "confidences"])


class FaceDetector(MainClass):
    def __init__(self, name, file_path, **kwargs):
        super().__init__(name, file_path=file_path, **kwargs)
        self.output_class = OUTPUT_CLASS

    @abstractmethod
    def detect_faces(self, img, is_rgb, confidence=None, get_time=False) -> OUTPUT_CLASS:
        pass
