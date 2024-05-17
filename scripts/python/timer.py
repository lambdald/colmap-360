import logging
from typing import Callable


def timer(func: Callable):
    def func_wrapper(*args, **kwargs):
        from time import time

        time_start = time()
        result = func(*args, **kwargs)
        time_end = time()
        time_spend = time_end - time_start
        logging.info(f"Function {func.__module__}.{func.__qualname__} cost time: {time_spend:.3f} s")
        return result

    return func_wrapper
