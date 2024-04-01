import base64
import struct
import logging
from typing import Union

import numpy as np

from deep_utils.utils.logging_utils.logging_utils import log_print, value_error_log


class BinaryUtils:

    @staticmethod
    def img_to_b64(image: np.ndarray, extension: str = ".jpg") -> str:
        """
        returns a base64 encoded string from an image
        :param image: The input image
        :param extension: the extension to encode
        :return: byte string
        """
        import base64

        import cv2

        _, encoded_img = cv2.imencode(extension, image)
        base64_img = base64.b64encode(encoded_img).decode("utf-8")
        return base64_img

    @staticmethod
    def b64_to_img(image_byte: Union[bytes, str]) -> np.ndarray:
        """
        Converts the input bytes or string to an 3-channel image
        :param image_byte: base64 image string
        :return: numpy image
        """
        import base64
        import cv2
        if isinstance(image_byte, bytes):
            image_byte = image_byte.decode()
        if ";" in image_byte:
            image_byte = image_byte.split(";")[-1]

        image_byte = image_byte.encode()
        im_bytes = base64.b64decode(image_byte)
        im_arr = np.frombuffer(im_bytes, dtype=np.uint8)  # im_arr is one-dim Numpy array
        img = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)

        return img

    @staticmethod
    def ndarray_to_b64(
            array: np.ndarray,
            dtype: Union[None, str, type] = None,
            append_shape=False,
            append_dtype=False,
            decode=None,
            logger: Union[None, logging.Logger] = None,
    ):
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
        shape_bytes = base64.struct.pack(
            f">6s{len(shape) + 1}I", bytes("shape:",
                                           "utf-8"), len(shape) * 4, *shape
        )
        array_bytes = base64.b64encode(array)
        if isinstance(dtype, str):
            dtype_name = dtype
        elif isinstance(dtype, type):
            dtype_name = dtype.__name__
        elif isinstance(dtype, np.dtype):
            dtype_name = dtype.name
        else:
            value_error_log(logger, f"dtype: {dtype} is not supported!")
        dtype_bytes = base64.struct.pack(
            f">6sI{len(dtype_name)}s",
            bytes("dtype:", "utf-8"),
            len(dtype_name),
            bytes(dtype_name, "utf-8"),
        )
        if not append_dtype and not append_dtype:
            res = array_bytes
        elif not append_dtype and append_shape:
            res = shape_bytes + array_bytes
        elif append_dtype and not append_shape:
            res = dtype_bytes + array_bytes
        elif append_dtype and append_shape:
            res = dtype_bytes + shape_bytes + array_bytes
        else:
            value_error_log(
                logger, f"dtype: {dtype}, shape: {shape} is not supported")

        if decode:
            res = res.decode(decode)

        return res

    @staticmethod
    def b64_to_ndarray(
            byte_array, dtype, shape, logger: Union[None, logging.Logger] = None, encode=None
    ):
        """
        Converting a base64 to ndarray. For images use "b64_to_img"
        :param byte_array:
        :param dtype: pass None if dtype is encoded in the byte_array
        :param shape: pass None if shape is encoded in the byte_array
        :param logger: logger instance
        :param encode: whether encode param or not!
        :return:
        """

        if encode:
            byte_array = byte_array.encode(encode)
        elif not encode and isinstance(byte_array, str):
            byte_array = byte_array.encode()

        if dtype is None and byte_array[:6].decode("utf-8") == "dtype:":
            dtype_len = struct.unpack(">I", byte_array[6:10])[0]
            dtype = byte_array[10: dtype_len + 10].decode("utf-8")
            byte_array = byte_array[dtype_len + 10:]
        elif dtype is None:
            dtype = np.float32
            log_print(
                logger,
                f"dtype is not defined in byte_array nor as an input, setting dtype to {dtype}",
            )
        elif isinstance(dtype, str):
            pass
        elif isinstance(dtype, type):
            dtype = str(dtype)
        else:
            value_error_log(logger, f"dtype: {dtype} is not supported!")

        if shape is None and byte_array[:6].decode("utf-8") == "shape:":
            shape_len = struct.unpack(">I", byte_array[6:10])[0]
            shape = struct.unpack(f">{shape_len // 4}I",
                                  byte_array[10: 10 + shape_len])
            byte_array = byte_array[shape_len + 10:]
        elif shape is None:
            log_print(logger, f"shape is not defined in byte_array nor as input")
        elif isinstance(shape, tuple) or isinstance(shape, list):
            pass
        else:
            value_error_log(logger, f"shape: {shape} is not supported!")
        if isinstance(byte_array, str):
            byte_array = bytes(byte_array, encoding="utf-8")
        ndarray = np.frombuffer(base64.decodebytes(byte_array), dtype=dtype)
        if shape is not None:
            ndarray = ndarray.reshape(shape)
        return ndarray
