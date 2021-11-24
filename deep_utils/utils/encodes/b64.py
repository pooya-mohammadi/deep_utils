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