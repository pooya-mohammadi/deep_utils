from .mtcnn_tf_face_detection import MTCNNTFFaceDetector
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) < 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
