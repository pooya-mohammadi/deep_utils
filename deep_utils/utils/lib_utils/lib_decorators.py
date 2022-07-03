import inspect
import time
from functools import wraps
from typing import Sequence, Iterable

import numpy as np


def get_from_config(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        config = self.config
        arguments = inspect.getfullargspec(func)
        if arguments.defaults is not None:
            kwargs_ = {
                i: j for i, j in zip(arguments.args[::-1], arguments.defaults[::-1])
            }
        else:
            kwargs_ = dict()
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
            if isinstance(in_, list) and np.all([len(input_obj.shape) == dim for input_obj in in_]):
                # if there is a list of inputs and each input is a ndarray with the shape of dim...
                return func(self, in_, *args, **kwargs)
            if len(in_.shape) == dim + 1:
                return func(self, in_, *args, **kwargs)
            elif len(in_.shape) == dim:
                in_ = np.expand_dims(in_, axis=0)
                results = func(self, in_, *args, **kwargs)
                if hasattr(results, "DictNamedTuple") and results.DictNamedTuple:
                    new_results = {
                        key: val[0] if val is not None and isinstance(val, Iterable) and len(val) == 1 else val
                        for key, val in results.items()
                    }
                    cls = type(results)
                    results = cls(**new_results)
                elif isinstance(results, tuple):
                    results = tuple(
                        [
                            res[0] if res is not None and len(
                                res) == 1 else res
                            for res in results
                        ]
                    )
                elif isinstance(results, dict):
                    results = {
                        key: val[0] if val is not None and len(
                            val) == 1 else val
                        for key, val in results.items()
                    }
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
        elapsed_time = kwargs.get("get_time", False)
        if elapsed_time:
            tic = time.time()
            results = func(self, *args, **kwargs)
            toc = time.time()
            elapsed_time = round(toc - tic, 4)
            if isinstance(results, dict):
                results["elapsed_time"] = elapsed_time
            elif hasattr(results, "DictNamedTuple") and results.DictNamedTuple:
                from deep_utils.utils.dict_named_tuple_utils import dictnamedtuple

                new_results = dict(results.items())
                new_results["elapsed_time"] = elapsed_time
                cls = dictnamedtuple(
                    results.TypeName, list(results._fields) + ["elapsed_time"]
                )
                results = cls(**new_results)
            else:
                results = tuple(list(results) + [elapsed_time])
            return results
        return func(self, *args, **kwargs)

    return wrapper


def rgb2bgr(in_):
    def inner_decorator(func):
        @wraps(func)
        def wrapper(self, in_img, *args, **kwargs):
            is_rgb = kwargs.get("is_rgb")
            # if not is_rgb and in_ == 'rgb':
            #     in_img = in_img[..., ::-1]
            # elif is_rgb and in_ == 'bgr':
            #     in_img = in_img[..., ::-1]
            # elif is_rgb and in_ == 'gray':
            #     in_img = np.dot(in_img[..., :3], [0.299, 0.587, 0.144])
            #     in_img = in_img.astype(np.uint8)
            # elif not is_rgb and in_ == 'gray':
            #     in_img = np.dot(in_img[..., :3][..., ::-1], [0.299, 0.587, 0.144])
            #     in_img = in_img.astype(np.uint8)
            in_img = lib_rgb2bgr(in_img, target_type=in_, is_rgb=is_rgb)
            return func(self, in_img, *args, **kwargs)

        return wrapper

    return inner_decorator


def lib_rgb2bgr(in_img, target_type, is_rgb):
    if not is_rgb and target_type == "rgb":
        in_img = in_img[..., ::-1]
    elif (is_rgb and target_type == "rgb") or (not is_rgb and target_type == 'bgr'):
        pass
    elif is_rgb and target_type == "bgr":
        in_img = in_img[..., ::-1]
    elif is_rgb and target_type == "gray":
        in_img = np.dot(in_img[..., :3], [0.299, 0.587, 0.144])
        in_img = in_img.astype(np.uint8)
    elif not is_rgb and target_type == "gray":
        in_img = np.dot(in_img[..., :3][..., ::-1], [0.299, 0.587, 0.144])
        in_img = in_img.astype(np.uint8)
    else:
        raise ValueError(
            f"[ERROR] The input target_type: {target_type} and is_rgb: {is_rgb} are not supported!"
        )
    return in_img


def in_shape_fix(in_: np.ndarray, size=4):
    if len(in_.shape) == size:
        return in_
    elif len(in_.shape) == size - 1:
        return np.expand_dims(in_, axis=0)
    else:
        raise ValueError(f"[Error] Input size: {in_.shape} is not valid!")


def out_shape_fix(results):
    if hasattr(results, "DictNamedTuple") and results.DictNamedTuple:
        new_results = {
            key: val[0]
            if val is not None and isinstance(val, Sequence) and len(val) == 1
            else val
            for key, val in results.items()
        }
        cls = type(results)
        results = cls(**new_results)
    elif isinstance(results, tuple):
        results = tuple(
            [
                val[0]
                if val is not None and isinstance(val, Sequence) and len(val) == 1
                else val
                for val in results
            ]
        )
    elif isinstance(results, dict):
        results = {
            key: val[0]
            if val is not None and isinstance(val, Sequence) and len(val) == 1
            else val
            for key, val in results.items()
        }
    else:
        results = results[0]
    return results


def cast_kwargs_dict(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        arguments = inspect.getfullargspec(func)
        for kwarg in arguments.args:
            if kwarg.endswith("_kwargs"):
                _kwargs = kwargs.get(kwarg, None)
                kwargs[kwarg] = dict() if _kwargs is None else _kwargs
        return func(self, *args, **kwargs)

    return wrapper
