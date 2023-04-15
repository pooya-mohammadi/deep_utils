try:
    from deep_utils._dummy_objects.vision.face_detection import UltralightTFFaceDetector
    from .ultralight_tf_face_detection import UltralightTFFaceDetector
except ModuleNotFoundError:
    pass
