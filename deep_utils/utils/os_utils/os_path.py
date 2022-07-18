import os
from typing import Union

IMG_EXTENSIONS = [
    ".jpg",
    ".JPG",
    ".jpeg",
    ".JPEG",
    ".png",
    ".PNG",
    ".ppm",
    ".PPM",
    ".bmp",
    ".BMP",
    ".tiff",
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def split_extension(path, extension: Union[str, None] = None,
                    suffix: Union[str, None] = None,
                    prefix: Union[str, None] = None,
                    artifact_type: Union[str, None] = None,
                    artifact_value: Union[str, int, None] = None,
                    extra_punctuation: Union[str, None] = None
                    ):
    """
    split extension or add new suffix, prefix, or extension
    :param path:
    :param extension:
    :param suffix:
    :param prefix:
    :param artifact_type: prefix or suffix
    :param artifact_value: value for defined prefix or suffix
    :param extra_punctuation: the punctuation to be used before suffix or after prefix
    :return:

    >>> split_extension("image.jpg", suffix="_res")
    'image_res.jpg'
    >>> split_extension("image.jpg")
    ('image', '.jpg')
    >>> split_extension("image.jpg", extension=".png")
    'image.png'
    >>> split_extension("image.jpg", extension="png")
    'image.png'
    >>> split_extension("image.jpg", extension="png", prefix="0_")
    '0_image.png'
    >>> split_extension("image.jpg", extension="png", suffix="_res", prefix="0_")
    '0_image_res.png'
    """
    artifact_value = str(artifact_value) if artifact_value is not None else None
    assert (artifact_value and artifact_type) or (
            not artifact_value and not artifact_type), "both artifact type and artifact value should be None or not None"

    if (artifact_value and artifact_type) and (suffix or prefix):
        raise ValueError("artifact value and artifact type cannot have values while suffix and prefix are not None")

    extra_punctuation = extra_punctuation if extra_punctuation else ""
    if artifact_type == "suffix":
        return _split_extension(path, extension=extension, suffix=extra_punctuation + artifact_value)
    elif artifact_type == "prefix":
        return _split_extension(path, extension=extension, prefix=artifact_value + extra_punctuation)
    elif artifact_type is None:
        return _split_extension(path, extension=extension, suffix=extra_punctuation + suffix if suffix else None,
                                prefix=prefix + extra_punctuation if prefix else None)
    else:
        raise ValueError(f"input artifact_type:{artifact_type} is invalid!")


def _split_extension(path, extension: Union[str, None] = None,
                     suffix: Union[str, None] = None,
                     prefix: Union[str, None] = None):
    """
    split extension or add new suffix, prefix, or extension
    :param path:
    :param extension:
    :param suffix:
    :param prefix:
    :return:

    >>> split_extension("image.jpg", suffix="_res")
    'image_res.jpg'
    >>> split_extension("image.jpg")
    ('image', '.jpg')
    >>> split_extension("image.jpg", extension=".png")
    'image.png'
    >>> split_extension("image.jpg", extension="png")
    'image.png'
    >>> split_extension("image.jpg", extension="png", prefix="0_")
    '0_image.png'
    >>> split_extension("image.jpg", extension="png", suffix="_res", prefix="0_")
    '0_image_res.png'
    """
    remain, extension_ = os.path.splitext(path)
    prefix = prefix if prefix else ""
    suffix = suffix if suffix else ""
    extension = extension if extension else ""

    extension = '.' + extension if extension and not extension.startswith(".") else extension
    core = prefix + remain + suffix

    if extension:
        return core + extension

    if not suffix and not prefix:
        return core, extension_
    else:
        return core + extension_


def split_all(path):
    all_parts = []
    while 1:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            all_parts.insert(0, parts[0])
            break
        elif parts[1] == path:  # sentinel for relative paths
            all_parts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            all_parts.insert(0, parts[1])
    return all_parts


def get_file_name(file):
    """
    Get the file's name
    :param file: the file in which code is running. Path in `__file__`
    :return:
    """
    return os.path.splitext(os.path.split(file)[-1])[0]
