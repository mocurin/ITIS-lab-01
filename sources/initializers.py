"""
Initizlizer functions package

Initializers are generators, which produce floats indefinetely.
Utilizing generators allows to produce values based on previously
produced values.
"""
from typing import Generator

# Type hints
Initializer = Generator[float, None, None]


def zeros() -> Initializer:
    """Zero initializer"""
    while True:
        yield 0.


def ones() -> Initializer:
    """One initializer"""
    while True:
        yield 1.
