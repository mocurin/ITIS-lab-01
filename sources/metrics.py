"""Common metric functions package"""
from typing import Any, Callable, List


# Type hinting
Metric = Callable[[List[float], List[int]], Any]


def hamming(result: List[float], target: List[float]) -> float:
    return sum(abs(x - y) for x, y in zip(result, target))
