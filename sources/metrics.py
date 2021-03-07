"""Common metric functions package"""
from typing import Any, Callable, List


# Type hinting
Metric = Callable[[List[float], List[float]], Any]


def hamming(result: List[float], target: List[float]):
    return sum(abs(x - y) for x, y in zip(result, target))
