from deep_utils.utils.lib_utils import import_module

UltralightTorchFaceDetector = import_module(
    'deep_utils.vision.face_detection.ultralight.torch.ultralight_torch_face_detection',
    'UltralightTorchFaceDetector')
UltralightTFFaceDetector = import_module(
    'deep_utils.vision.face_detection.ultralight.tf.ultralight_tf_face_detection',
    'UltralightTFFaceDetector')

