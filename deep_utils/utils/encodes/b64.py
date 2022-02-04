import logging
from typing import Union
from deep_utils.utils.utils.logging_ import log_print, value_error_log
import numpy as np


def img_to_b64(image: np.ndarray, extension: str = '.jpg') -> str:
    """
    returns a base64 encoded string from an image
    :param image: The input image
    :param extension: the extension to encode
    :return: byte string
    """
    import base64
    import cv2
    _, encoded_img = cv2.imencode(extension, image)
    base64_img = base64.b64encode(encoded_img).decode('utf-8')
    return base64_img


def b64_to_img(image_string: str) -> np.ndarray:
    """
    Converts the input byte string to an RGB image
    :param image_string: base64 image string
    :return: numpy image
            """
    import cv2
    import base64

    img_data = base64.b64decode(image_string)
    image = np.array(bytearray(img_data), dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    return image


def ndarray_to_b64(array: np.ndarray,
                   dtype: Union[None, tuple] = None,
                   append_shape=True,
                   append_dtype=True,
                   utf_8_decode=True,
                   logger: Union[None, logging.Logger] = None):
    """
    Converting a ndarray to base64. For images use "img_to_b64"
    :param array:
    :param dtype:
    :return:
    """
    import base64
    if dtype is None:
        dtype = array.dtype
    array = array.astype(dtype)
    shape = array.shape
    shape_bytes = base64.struct.pack(f'>6s{len(shape) + 1}I', bytes("shape:", "utf-8"), len(shape) * 4, *shape)
    array_bytes = base64.b64encode(array)
    dtype_name = dtype.name
    dtype_bytes = base64.struct.pack(f">6sI{len(dtype_name)}s", bytes("dtype:", "utf-8"), len(dtype_name),
                                     bytes(dtype_name, "utf-8"))
    if not append_dtype and not append_dtype:
        res = array_bytes
    elif not append_dtype and append_shape:
        res = shape_bytes + array_bytes
    elif append_dtype and not append_shape:
        res = dtype_bytes + array_bytes
    elif append_dtype and append_shape:
        res = dtype_bytes + shape_bytes + array_bytes
    else:
        value_error_log(logger, f"dtype: {dtype}, shape: {shape} is not supported")

    if utf_8_decode:
        res = res.decode("utf-8")

    return res


def b64_to_ndarray(byte_array, dtype=None, shape=None, logger: Union[None, logging.Logger] = None, encdoe=True):
    """
    Converting a base64 to ndarray. For images use "b64_to_img"
    :param byte_array:
    :param dtype:
    :param shape:
    :param logger:
    :return:
    """
    import struct
    import base64

    if encdoe:
        byte_array = byte_array.encode('utf-8')

    if byte_array[:6].decode('utf-8') == "dtype:":
        dtype_len = struct.unpack(">I", byte_array[6: 10])[0]
        dtype = byte_array[10:dtype_len + 10].decode("utf-8")
        byte_array = byte_array[dtype_len + 10:]
    elif dtype is None:
        dtype = np.float32
        log_print(logger, f"dtype is not defined in byte_array nor as an input, setting dtype to {dtype}")
    else:
        value_error_log(logger, f"dtype: {dtype} is not supported!")

    if byte_array[:6].decode('utf-8') == "shape:":
        shape_len = struct.unpack(">I", byte_array[6: 10])[0]
        shape = struct.unpack(f">{shape_len // 4}I", byte_array[10:10 + shape_len])
        byte_array = byte_array[shape_len + 10:]
    elif shape is None:
        log_print(logger, f"shape is not defined in byte_array nor as input")
    else:
        value_error_log(logger, f"shape: {shape} is not supported!")

    ndarray = np.frombuffer(base64.decodebytes(byte_array), dtype=dtype)
    if shape is not None:
        ndarray = ndarray.reshape(shape)
    return ndarray


# if __name__ == '__main__':
#     array = np.random.random((20, 5)).astype(np.float32)
#     res = ndarray_to_b64(array)
#     new_array = b64_to_ndarray(res)
#     np.all(array == new_array)
