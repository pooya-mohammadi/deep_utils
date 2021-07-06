from deep_utils.utils.lib_utils import import_module

MTCNNTorchFaceDetector = import_module('deep_utils.vision.face_detection.mtcnn_.caffe.mtcnn_torch_face_detection',
                                       'MTCNNTorchFaceDetector')
MTCNNTFFaceDetector = import_module('deep_utils.vision.face_detection.mtcnn_.tf.mtcnn_tf_face_detection',
                                    'MTCNNTFFaceDetector')
