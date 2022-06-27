import cv2
import pytest

from deep_utils import Box, UltralightTorchFaceDetector, download_file


@pytest.mark.torchvision
def test_ultralight_slim():
    model = UltralightTorchFaceDetector(device="cpu", model_name="slim")
    file_path = download_file(
        "https://github.com/pooya-mohammadi/deep-utils-notebooks/releases/download/0.1.0/movie-stars.jpg"
    )
    img = cv2.imread(file_path)
    res = model.detect_faces(img, is_rgb=False, get_time=True)
    assert len(res["boxes"]) >= 0, "Model did not detect anything!"
    img = Box.put_box(img, res["boxes"])
    assert img is not None, "put_box returned nothing!"


@pytest.mark.torchvision
def test_ultralight_rbf():
    model = UltralightTorchFaceDetector(device="cpu", model_name="RBF")
    file_path = download_file(
        "https://github.com/pooya-mohammadi/deep-utils-notebooks/releases/download/0.1.0/movie-stars.jpg"
    )
    img = cv2.imread(file_path)
    res = model.detect_faces(img, is_rgb=False, get_time=True)
    assert len(res["boxes"]) >= 0, "Model did not detect anything!"
    img = Box.put_box(img, res["boxes"])
    assert img is not None, "put_box returned nothing!"
