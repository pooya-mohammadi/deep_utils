import numpy as np
from deep_utils.utils.box_utils.boxes import Point


def resize(img, dsize, in_source='Numpy', mode='cv2', interpolation=None):
    mode = 'cv2' if mode is None else mode
    if mode == 'cv2':
        dsize = Point.point2point(dsize, in_source=in_source, to_source=Point.PointSource.Numpy)
        new_img = cv2_resize(img, dsize=dsize, interpolation=interpolation)
    else:
        raise ValueError(f'{mode} is not supported, supported types: cv2, ')
    return new_img


def cv2_resize(img, dsize, dst=None, fx=None, fy=None, interpolation=None):
    import cv2
    if len(img.shape) == 3:
        return cv2.resize(img, dsize, dst, fx, fy, interpolation)
    elif len(img.shape) == 4:
        return np.array([cv2.resize(im, dsize, dst, fx, fy, interpolation) for im in img])


def get_img_shape(img):
    if len(img.shape) == 4:
        return img.shape
    elif len(img.shape) == 3:
        return np.expand_dims(img, axis=0).shape
    else:
        raise Exception(f'shape: {img.shape} is not an image')
