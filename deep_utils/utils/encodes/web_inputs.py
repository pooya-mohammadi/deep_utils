import numpy as np


def read_img_form(contents, ret_rgb: bool = False):
    """
    read input from a form request with cv2.
    :param contents: The parsed form input
    :param ret_rgb: If set true, will change the output to rgb
    :return: cv2 - img
    """
    import cv2

    image = np.array(bytearray(contents.read()), dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    if ret_rgb:
        # convert to rgb
        image = image[..., ::-1]
    return image
