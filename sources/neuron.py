"""Neuron class implementation package"""
from itertools import cycle
from typing import List

from .activations import Identity, Activation
from .initializers import zeros, ones, Initializer


class Neuron:
    def __init__(self,
                 inputs: int,
                 *,
                 use_bias: bool = True,
                 activation: Activation = Identity(),
                 weights_initializer: Initializer = zeros,
                 bias_initializer: Initializer = ones):
        self._weights = [weight for _, weight in zip(range(inputs), weights_initializer())]
        self._bias = 0.

        if use_bias:
            self._bias = bias_initializer()

        self._activation = activation

    def __call__(self, samples: List[float]) -> float:
        if len(samples) != len(self._weights):
            raise ValueError(f"Input size and weights size must be equal:"
                             f"{len(samples)} vs {len(self._weights)}")

        net = sum(value * weight
                  for value, weight
                  in zip(samples, self._weights))

        net += self._bias

        return self._activation(net)

    def _fit_once(self, samples: List[float], target: float, norm: float) -> float:
        net = self(samples)

        delta = target - net

        self._weights = [weight + norm * delta * self._activation.derivative(output_value) * input_value
                         for output_value, input_value, weight
                         in zip(net, samples, self._weights)]

        return delta

    def fit(self, samples: List[List[float]], targets: List[float], norm: float, *, verbose: bool = True):
        if len(samples) != len(targets):
            raise ValueError(f"Learning data shapes mismatch: {len(samples)} vs {len(targets)}")

        for samples, target in cycle(zip(samples, targets)):
            error = self._fit_once(samples, target)

            if not error:
                return
