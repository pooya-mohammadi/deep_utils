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


def b64_to_img(image_string: str, is_rgb: bool = True) -> np.ndarray:
    """
    Converts the input byte string to an RGB image
    :param image_string: base64 image string
    :param is_rgb: whether the input string is made from an rgb image or bgr.
    :return: numpy image
            """
    import cv2
    import base64
    import io
    from PIL import Image

    img_data = base64.b64decode(image_string)
    image = Image.open(io.BytesIO(img_data))
    image = np.array(image)
    if not is_rgb:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image
