"""Common activation functions package"""
import abc

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


class Identity(IActivation):
    def __call__(self, value: float) -> float:
        return value

    def derivative(self, _) -> float:
        return 1.


identity = Identity()
