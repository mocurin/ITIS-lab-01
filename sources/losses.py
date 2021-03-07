"""Common loss functions package"""
from typing import Callable


# Type hinting
Loss = Callable[[float, float], float]


def difference(result: float, target: float) -> float:
    return target - result
