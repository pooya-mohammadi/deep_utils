import pytest

from deep_utils import ndarray_to_b64, b64_to_ndarray


@pytest.mark.basic
def test_ndarray_to_b64_to_ndarray():
    """
    Testing ndarray_to_b64 and b64_to_ndarray
    Returns: Nothing

    """
    import numpy as np
    array = np.random.random((20, 5)).astype(np.float32)
    res = ndarray_to_b64(array, decode="utf-8")
    new_array = b64_to_ndarray(
        res, dtype="float32", shape=(20, 5), encode="utf-8")
    assert np.all(array == new_array), "with shapes and dtype does not work"

    res = ndarray_to_b64(array, append_shape=True, append_dtype=True)
    new_array = b64_to_ndarray(res, dtype=None, shape=None)
    assert np.all(array == new_array), "with appended shapes and dtype does not work"

    res = ndarray_to_b64(array, append_shape=True, append_dtype=True)
    new_array = b64_to_ndarray(res.decode(), dtype=None, shape=None)
    assert np.all(array == new_array), "with input str and appended shapes and dtype does not work"
