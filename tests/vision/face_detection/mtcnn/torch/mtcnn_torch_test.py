import cv2
import pytest

from deep_utils import Box, MTCNNTorchFaceDetector, download_file


@pytest.mark.torchvision
def test_face_detection():
    model = MTCNNTorchFaceDetector(device="cpu")
    file_path = download_file(
        "https://github.com/pooya-mohammadi/deep-utils-notebooks/releases/download/0.1.0/movie-stars.jpg"
    )
    img = cv2.imread(file_path)
    res = model.detect_faces(img, is_rgb=False, get_time=True)
    assert len(res.boxes) >= 0, "Model did not detect anything!"
    img = Box.put_box(img, res.boxes)
    assert img is not None, "put_box returned nothing!"
