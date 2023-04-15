try:
    from deep_utils._dummy_objects.vision.face_detection import UltralightTorchFaceDetector
    from .ultralight_torch_face_detection import UltralightTorchFaceDetector
except ModuleNotFoundError:
    pass
