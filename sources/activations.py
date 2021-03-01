"""
Activation functions package

Activations are functions, which take float value and apply activation
function to it, producing another float value
"""
from typing import Callable

# Type hints
Activation = Callable[[float], float]


class IActivation:
    def __call__(self, value: float) -> float:
        raise NotImplementedError

    def derivative(self, value: float) -> float:
        raise NotImplementedError


class Identity(IActivation):
    def __call__(self, value: float) -> float:
        return value

    def derivative(self, _) -> float:
        return 1.
