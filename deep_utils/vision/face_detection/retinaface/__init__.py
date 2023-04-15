try:
    from deep_utils._dummy_objects.vision.face_detection import RetinaFaceTorchFaceDetector
    from .torch.retinaface_torch_face_detection import RetinaFaceTorchFaceDetector
except ModuleNotFoundError:
    pass
