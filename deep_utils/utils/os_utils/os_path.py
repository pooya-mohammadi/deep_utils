import os
from pathlib import Path
from typing import Union, List, Tuple, Optional

IMG_EXTENSIONS = (
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
)


def validate_file_extension(filename, extensions: Union[List[str], Tuple[str]]):
    """
    validates a file with provided extensions. It checks the extension in both uppercase and lowercase mode.
    :param filename:
    :param extensions:
    :return:
    """
    return any(filename.endswith(extension.upper()) or filename.endswith(extension.lower()) for extension in extensions)


def is_img(filename, extensions=IMG_EXTENSIONS):
    """
    validate whether the input filename is img or not
    :param filename:
    :param extensions:
    :return:
    >>> is_img("file1.png")
    True
    >>> is_img("file1.jPg")
    False
    >>> is_img("file1.wow")
    False
    """
    return validate_file_extension(filename, extensions)


def split_extension(path, extension: Union[str, None] = None,
                    suffix: Union[str, None] = None,
                    prefix: Union[str, None] = None,
                    artifact_type: Union[str, None] = None,
                    artifact_value: Union[str, int, None] = None,
                    extra_punctuation: Union[str, None] = None,
                    current_extension: Optional[str] = None
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
    :param current_extension: the current extension of the file.
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
    >>> split_extension("image.nii.gz", current_extension=".nii.gz", suffix="_crop")
    'image_crop.nii.gz'
    """
    artifact_value = str(artifact_value) if artifact_value is not None else None
    assert (artifact_value and artifact_type) or (
            not artifact_value and not artifact_type), \
        "both artifact type and artifact value should be None or not None"

    if (artifact_value and artifact_type) and (suffix or prefix):
        raise ValueError("artifact value and artifact type cannot have values while suffix and prefix are not None")

    extra_punctuation = extra_punctuation if extra_punctuation else ""
    if artifact_type == "suffix":
        return _split_extension(path, extension=extension, suffix=extra_punctuation + artifact_value,
                                current_extension=current_extension)
    elif artifact_type == "prefix":
        return _split_extension(path, extension=extension, prefix=artifact_value + extra_punctuation,
                                current_extension=current_extension)
    elif artifact_type is None:
        return _split_extension(path, extension=extension, suffix=extra_punctuation + suffix if suffix else None,
                                prefix=prefix + extra_punctuation if prefix else None,
                                current_extension=current_extension)
    else:
        raise ValueError(f"input artifact_type:{artifact_type} is invalid!")


def _split_extension(path: Union[Path, str],
                     extension: Union[str, None] = None,
                     suffix: Union[str, None] = None,
                     prefix: Union[str, None] = None,
                     current_extension: Optional[str] = None
                     ):
    """
    split extension or add new suffix, prefix, or extension
    :param path:
    :param extension:
    :param suffix:
    :param prefix:
    :param current_extension: the current extension of the file.
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
    >>> split_extension("image.nii.gz", current_extension=".nii.gz", suffix="_crop")
    'image_crop.nii.gz'
    """
    if current_extension:
        if path.endswith(current_extension):
            remain = path[:-len(current_extension)]
            extension_ = current_extension
        else:
            raise ValueError(f"current extension:{current_extension} is not at the of the path:{path}")
    else:
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
    """
    splits and extracts all the parts of an input path
    :param path:
    :return:
    >>> split_all(r"Users\pooya\projects\deep_utils")
    ['Users', 'pooya', 'projects', 'deep_utils']
    """
    all_parts = []
    while True:
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
