try:
    from deep_utils.dummy_objects.vision.face_detection import RetinaFaceTorchFaceDetector
    from .torch.retinaface_torch_face_detection import RetinaFaceTorchFaceDetector
except ModuleNotFoundError:
    pass
