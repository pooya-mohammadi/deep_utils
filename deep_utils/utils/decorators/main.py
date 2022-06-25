import time
from functools import wraps
from typing import Callable


def get_method_time(method) -> Callable:
    """
    Gets the elapsed time of a method
    :param method: The method that will be measured
    :return:
    """

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        tic = time.time()
        results = method(self, *args, **kwargs)
        toc = time.time()
        elapsed_time = round(toc - tic, 4)
        print(
            f"elapsed time for {self.__class__.__name__}.{method.__name__}: {elapsed_time}"
        )
        return results

    return wrapper


def get_func_time(func) -> Callable:
    """
    Gets the elapsed time of a class
    :param func: The func that will be measured
    :return:
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        tic = time.time()
        results = func(*args, **kwargs)
        toc = time.time()
        elapsed_time = round(toc - tic, 4)
        print(f"elapsed time for {func.__name__}: {elapsed_time} seconds")
        return results

    return wrapper
