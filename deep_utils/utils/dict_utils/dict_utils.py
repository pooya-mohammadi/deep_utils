class frozendict:
    def __init__(self, **kwargs):
        super(frozendict, self).__init__()
        self.repo = dict()
        for key, item in kwargs.items():
            self.repo[key] = item

    def __getitem__(self, item):
        return self.repo[item]

    def __setitem__(self, key, value):
        raise TypeError(f"frozendict object does not support updating")


def get_dict_extreme(dict_: dict, key, mode='max', list_values=False, ignore_non_sequences=True):
    """
    Gets the maximum/minimum item of a dict based on a key in the input dict
    :param dict_:
    :param key:
    :param mode:
    :param list_values: whether list values, so it would have the same shape as the input
    :param ignore_non_sequences: In the input dict, there may be some features which are not a sequence type and do no
    accept index. Ignore if those samples are seen.
    :return:
    """
    import numpy as np
    if mode == 'max':
        index = np.argmax(dict_[key])
    elif mode == 'min':
        index = np.argmax(dict_[key])
    else:
        raise ValueError()
    res = dict()

    for k, v in dict_.items():
        try:
            value = v[index]
        except TypeError as t:
            if ignore_non_sequences:
                value = v
            else:
                raise TypeError(t)
        if list_values:
            res[k] = [value]
        else:
            res[k] = value

    return res
