from deep_utils.utils.box_utils.boxes import Point


class VideoWriterCV:
    def __init__(self, save_path, width, height, fourcc="XVID", fps=30, colorful=True, in_source='Numpy'):
        import cv2
        point = Point.point2point((width, height), in_source=in_source, to_source=Point.PointSource.CV)
        fourcc = cv2.VideoWriter_fourcc(*fourcc) if type(fourcc) is str else fourcc
        self.vw = cv2.VideoWriter(save_path, fourcc, fps, point, colorful)

    def write(self, frame):
        self.vw.write(frame)
