import warnings

import pytest

warnings.filterwarnings("ignore")


@pytest.mark.basic
def test_resize_ratio():
    """
    Testing the resize_ratio function
    Returns: Nothing

    """
    import numpy as np

    from deep_utils import resize_ratio

    dummy_images = [
        np.random.randint(0, 255, (1200, 900, 3), dtype=np.uint8),
        np.random.randint(0, 255, (800, 1200, 3), dtype=np.uint8),
        np.random.randint(0, 255, (550, 350, 3), dtype=np.uint8),
        np.random.randint(0, 255, (900, 900, 3), dtype=np.uint8),
    ]
    preferred_outputs = [
        (900, 675, 3),
        (600, 900, 3),
        (900, 572, 3),
        (900, 900, 3),
    ]

    for dummy_img, preferred_output in zip(dummy_images, preferred_outputs):
        out = resize_ratio(dummy_img, 900)
        assert (
            out.shape == preferred_output
        ), f"resize_ratio failed for input img.shape={dummy_img.shape}"
