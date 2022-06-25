import numpy as np


def shuffle_group(array_a: np.ndarray, array_b: np.ndarray = None, copy=False):
    """
    :param array_a: The input array. It could be a batch of images or ...
    :param array_b: The labels or the masks of the input array that should be shuffled together.
    :param copy: whether generate a copy of the inputs and apply the function or not
    :return:
    returns the shuffle format of array a and b if the latter one exists.
    """
    if copy:
        array_a = array_a.copy()
    if array_b is not None:
        if copy:
            array_b = array_b.copy()
        indices = np.arange(len(array_a))
        np.random.shuffle(indices)
        array_a[:] = array_a[indices]
        array_b[:] = array_b[indices]
        return array_a, array_b
    else:
        np.random.shuffle(array_a)
        return array_a


def shuffle_group_torch(array_a, array_b=None, copy=False):
    """
    :param array_a: The input array. It could be a batch of images or ...
    :param array_b: The labels or the masks of the input array that should be shuffled together.
    :param copy: whether generate a copy of the inputs and apply the function or not
    :return:
    returns the shuffle format of array a and b if the latter one exists.
    """
    import torch

    if copy:
        array_a = torch.clone(array_a)
    if array_b is not None:
        if copy:
            array_b = torch.clone(array_b)
        indices = np.arange(len(array_a))
        np.random.shuffle(indices)
        array_a[np.arange(len(array_a))] = array_a[indices]
        array_b[np.arange(len(array_a))] = array_b[indices]
        return array_a, array_b
    else:
        indices = np.arange(len(array_a))
        np.random.shuffle(indices)
        array_a[np.arange(len(array_a))] = array_a[indices]
        return array_a
