__author__ = "Alex Rogozhnikov"
__version__ = "0.3.2"


class EinopsError(RuntimeError):
    """Runtime error thrown by einops"""

    pass


__all__ = ["rearrange", "reduce", "repeat", "parse_shape", "asnumpy", "EinopsError"]

from .einops import asnumpy, parse_shape, rearrange, reduce, repeat
