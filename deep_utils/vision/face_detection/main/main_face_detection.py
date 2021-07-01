from deep_utils.main_abs.main import MainClass
from abc import abstractmethod


class FaceDetector(MainClass):
    def __init__(self):
        super().__init__()
        self.model = None

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def detect_faces(self, img):
        pass
