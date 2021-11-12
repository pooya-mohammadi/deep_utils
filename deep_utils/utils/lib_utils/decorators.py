import time
from functools import wraps


def get_method_time(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        tic = time.time()
        results = func(self, *args, **kwargs)
        toc = time.time()
        elapsed_time = round(toc - tic, 4)
        print(f"elapsed time for {self.__class__.__name__}.{func.__name__}: {elapsed_time}")
        return results

    return wrapper
