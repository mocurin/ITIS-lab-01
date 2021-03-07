"""Common initializer functions package"""
from typing import Generator

# Type hinting
Initializer = Generator[float, None, None]


def zeros() -> Initializer:
    """Zero initializer"""
    while True:
        yield 0.


def ones() -> Initializer:
    """One initializer"""
    while True:
        yield 1.
