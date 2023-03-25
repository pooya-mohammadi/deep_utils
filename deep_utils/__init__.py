from typing import TYPE_CHECKING

from deep_utils.dummy_objects.dummy_framework.dummy_framework import _LazyModule
from deep_utils.dummy_objects.dummy_framework.dummy_framework import (
    is_torch_available,
    is_tf_available,
    is_transformers_available,
    is_cv2_available,
    is_torchvision_available,
)

# Deep Utils version number
__version__ = "1.0.0"

# no third-party python libraries are required for the following classes
_import_structure = {
    "utils.box_utils": ["Box", "Point"],
}

if is_torch_available() and is_cv2_available():
    _import_structure["vision.face_detection.mtcnn.torch.mtcnn_torch_face_detection"] = ["MTCNNTorchFaceDetector"]
else:
    from .dummy_objects import torch_cv2_dummy
    _import_structure["dummy_objects.torch_cv2_dummy"] = [
        name for name in dir(torch_cv2_dummy) if not name.startswith("_")
    ]

if is_tf_available() and is_cv2_available():
    _import_structure["vision.face_detection.mtcnn.tf.mtcnn_tf_face_detection"] = ["MTCNNTFFaceDetector"]
else:
    from .dummy_objects import tf_cv2_dummy

    _import_structure["dummy_objects.tf_cv2_dummy"] = [
        name for name in dir(tf_cv2_dummy) if not name.startswith("_")
    ]

if TYPE_CHECKING:
    from .utils.box_utils import Box, Point
    from .vision.face_detection.haarcascade.cv2_.haarcascade_cv2_face_detection import HaarcascadeCV2FaceDetector
    from .vision.face_detection.mtcnn.tf.mtcnn_tf_face_detection import MTCNNTFFaceDetector
    from .vision.face_detection.mtcnn.torch.mtcnn_torch_face_detection import MTCNNTorchFaceDetector

else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
        extra_objects={"__version__": __version__},
    )
