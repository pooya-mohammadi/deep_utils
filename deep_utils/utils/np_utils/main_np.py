import numpy as np

from deep_utils.utils.logging_utils import log_print, value_error_log


def repeat_dimension(input_array: np.ndarray, n=3, d=2, logger=None, verbose=0):
    """

    Args:
        input_array:
        n: how many times to repeat
        d: dimension to be repeated!
        logger:
        verbose: whether print or not

    Returns:

    """
    if not isinstance(input_array, np.ndarray):
        value_error_log(logger, "input_array is not of type array")
    elif len(input_array.shape) == d:
        input_array = np.expand_dims(input_array, axis=-1)
    elif len(input_array.shape) < d:
        value_error_log(
            logger,
            f"input_array.shape:{input_array.shape} is lower the one:{d} which should be repeated!",
        )
    result = np.repeat(input_array, n, axis=d)
    log_print(
        logger, f"Successfully create result.shape: {result.shape}!", verbose=verbose
    )
    return result
