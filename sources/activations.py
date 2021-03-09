"""Common activation functions package"""
import abc
import math

from .misc import subclasshook_helper


REQUIRED = (
    '__call__',
    'derivative'
)


class IActivation(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass) -> bool:
        return subclasshook_helper(REQUIRED)

    @abc.abstractmethod
    def __call__(self, value: float) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def derivative(self, value: float) -> float:
        raise NotImplementedError


# Type hinting alias
Activation = IActivation


# str to Activation mapping
activations = dict()


def register(cls):
    """Register class in activations dictionary"""
    activations[cls.__name__] = cls

    return cls


@register
class Identity(IActivation):
    def __call__(self, value: float) -> float:
        return value

    def derivative(self, _) -> float:
        return 1.


identity = Identity()


@register
class Threshold(IActivation):
    def __init__(self, threshold: float = 0):
        self._threshold = threshold

    def __call__(self, value: float) -> float:
        return 1. if value >= self._threshold else 0.

    def derivative(self, _) -> float:
        return 1.


@register
class Logistic(IActivation):
    def __call__(self, value: float) -> float:
        return 1. / (1. + math.exp(-value))

    def derivative(self, value: float) -> float:
        return self(value) * (1. - self(value))
