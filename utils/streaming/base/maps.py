import functools
import traceback

from pybeamline.stream.base_map import BaseMap


def catch_and_reraise(method):
    """Decorator for catching exceptions in `transform`."""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except Exception:
            print(f'[ERROR] {self.__class__.__name__}.{method.__name__} crashed')
            traceback.print_exc()
            raise

    return wrapper


class EmptyMap(BaseMap):
    def transform(self, item):
        return [item]
