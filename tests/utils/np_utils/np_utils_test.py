import pytest

from deep_utils import repeat_dimension


@pytest.mark.basic
def test_repeat_dimension():
    """
    Testing the repeat dimension function
    Returns: Nothing

    """
    import numpy as np

    dummy_arrays = [
        np.random.randint(0, 255, (1200, 900, 1), dtype=np.uint8),
        np.random.randint(0, 255, (800, 1200), dtype=np.uint8),
    ]

    preferred_outputs = [
        (1200, 900, 3),
        (800, 1200, 3),
    ]

    for dummy_array, preferred_output in zip(dummy_arrays, preferred_outputs):
        out = repeat_dimension(dummy_array, n=3, d=2)
        assert (
            out.shape == preferred_output
        ), f"repeat_dimension failed for input array.shape={dummy_array.shape} returned: {out.shape}"

    # failures
    with pytest.raises(Exception) as e:
        # size failure
        repeat_dimension(np.random.randint(0, 255, (800,)), d=2)
        # type failure
        repeat_dimension(10, d=2)
