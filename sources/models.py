from itertools import cycle
from typing import List

from .activations import Identity, Activation
from .initializers import zeros, ones, Initializer
from .historian import Historian


class IModel:
    def __init__(self):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def fit(self, *args, **kwargs):
        raise NotImplementedError

    def predict(self, *args, **kwargs):
        raise NotImplementedError


class Neuron:
    def __init__(self,
                 inputs: int,
                 *,
                 use_bias: bool = True,
                 activation: Activation = Identity(),
                 weights_initializer: Initializer = zeros,
                 bias_initializer: Initializer = ones):
        self._weights = [weight for _, weight in zip(range(inputs), weights_initializer())]
        self._bias = None

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

        if self._bias is not None:
            net += self._bias

        return self._activation(net)

    def _fit_once(self, samples: List[float], target: float, norm: float) -> float:
        net = self(samples)

        delta = target - net

        self._weights = [weight + norm * delta * self._activation.derivative(net) * input_value
                         for input_value, weight
                         in zip(samples, self._weights)]

        if self._bias is not None:
            self._bias += norm * delta * self._activation.derivative(net)

        return delta

    def fit(self, samples: List[List[float]], targets: List[float], norm: float, *, verbose: bool = True) -> List:
        if len(samples) != len(targets):
            raise ValueError(f"Learning data shapes mismatch: {len(samples)} vs {len(targets)}")

        historian = Historian('Epoch: {}, sample: {}, error: {}, weights: {}, bias: {}' if verbose else '')

        epoch_size = len(samples)

        zipped_data = zip(samples, targets)
        endless_data_generator = cycle(zipped_data)
        data_enumerator = enumerate(endless_data_generator)

        for idx, (samples, target) in data_enumerator:
            error = self._fit_once(samples, target)

            historian.append(idx // epoch_size,
                             idx % epoch_size,
                             error,
                             self._weights,
                             self._bias)

            if not error:
                return
