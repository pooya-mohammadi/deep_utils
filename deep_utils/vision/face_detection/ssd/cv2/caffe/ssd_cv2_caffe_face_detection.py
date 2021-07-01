from deep_utils.main_abs.cv2.cv2_caffe import CV2Caffe


class SSDCV2CaffeFaceDetector(CV2Caffe):

    def __init__(self, **kwargs):
        super().__init__(name=self.__class__.__name__, file_path=__file__, **kwargs)


if __name__ == '__main__':
    model = SSDCV2CaffeFaceDetector()
