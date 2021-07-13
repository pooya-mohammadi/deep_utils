import inspect
import time
from functools import wraps

import numpy as np


def get_from_config(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        config = self.config
        arguments = inspect.getfullargspec(func)
        kwargs_ = {i: j for i, j in zip(arguments.args[::-1], arguments.defaults[::-1])}
        if kwargs is not None:
            kwargs_.update(kwargs)
        kwargs = kwargs_
        for key, val in kwargs.items():
            if val is None and hasattr(config, key):
                val = getattr(config, key)
                kwargs[key] = val
        return func(self, *args, **kwargs)

    return wrapper


def expand_input(dim):
    def inner_decorator(func):
        @wraps(func)
        def wrapper(self, in_, *args, **kwargs):
            if len(in_.shape) == dim + 1:
                return func(self, in_, *args, **kwargs)
            elif len(in_.shape) == dim:
                in_ = np.expand_dims(in_, axis=0)
                results = func(self, in_, *args, **kwargs)
                if type(results) is tuple:
                    results = tuple([res[0] if res is not None else res for res in results])
                else:
                    results = results[0]
                return results
            else:
                raise Exception(f"shape {in_.shape} is not supported.")

        return wrapper

    return inner_decorator


def get_elapsed_time(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        elapsed_time = kwargs.get('get_time', False)
        if elapsed_time:
            tic = time.time()
            results = func(self, *args, **kwargs)
            toc = time.time()
            return tuple(list(results) + [round(toc - tic, 4)])
        return func(self, *args, **kwargs)

    return wrapper


def rgb2bgr(in_):
    def inner_decorator(func):
        @wraps(func)
        def wrapper(self, in_img, *args, **kwargs):
            is_rgb = kwargs.get('is_rgb', False)
            if not is_rgb and in_ == 'rgb':
                in_img = in_img[..., ::-1]
            elif is_rgb and in_ == 'bgr':
                in_img = in_img[..., ::-1]

            return func(self, in_img, *args, **kwargs)

        return wrapper

    return inner_decorator
