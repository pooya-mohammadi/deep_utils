from deep_utils.dummy_objects.dummy_framework import DummyObject, requires_backends


class MTCNNTFFaceDetector(metaclass=DummyObject):
    _backend = ["tf", "cv2"]
    _module = "deep_utils.vision.face_detection.mtcnn.tf.mtcnn_tf_face_detection"

    def __init__(self, *args, **kwargs):
        requires_backends(self, self._backend, module_name=self._module, cls_name=self.__class__.__name__)
