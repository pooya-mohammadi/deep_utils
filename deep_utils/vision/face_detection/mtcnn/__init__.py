from deep_utils.utils.lib_utils import import_module
from .torch import MTCNNTorchFaceDetector

MTCNNTFFaceDetector = import_module('deep_utils.vision.face_detection.mtcnn.tf.mtcnn_tf_face_detection',
                                    'MTCNNTFFaceDetector')
