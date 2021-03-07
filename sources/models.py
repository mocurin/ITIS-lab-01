"""Models package"""
import abc

from typing import Any, Callable, Dict, Iterable, List, Tuple

from .activations import identity, Activation
from .generators import DataGenerator
from .initializers import zeros, ones, Initializer
from .historian import Historian
from .losses import difference, Loss
from .metrics import hamming, Metric
from .misc import subclasshook_helper


REQUIRED = (
    '__call__',
    'fit',
    'predict'
)


class IModel(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass) -> bool:
        return subclasshook_helper(REQUIRED)

    @abc.abstractmethod
    def fit(self, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, *args, **kwargs):
        raise NotImplementedError


# Type hinting alias
Model = IModel
MetricEarlyStop = Callable[[Any], bool]

DEFAULT_METRICS = {
    'Hamming distance': hamming
}

DEFAULT_METRICS_EARLY_STOP = {
    'Hamming distance': lambda x: x != 0
}


class Neuron(IModel):
    def __init__(self,
                 inputs: int,
                 *,
                 use_bias: bool = True,
                 activation: Activation = identity,
                 weights_initializer: Initializer = zeros(),
                 bias_initializer: Initializer = ones(),
                 loss: Loss = difference,
                 metrics: Dict[str, Metric] = None,
                 metrics_early_stop: Dict[str, MetricEarlyStop] = None):
        # Generate weights by taking `len(inputs)` times from `weights_initializer`
        self._weights = [weight
                         for _, weight
                         in zip(inputs, weights_initializer)]

        # Take once from bias generator
        self._bias = next(bias_initializer) if use_bias else None

        # Avoid mutable default argument
        self._metrics = DEFAULT_METRICS if metrics is None else metrics

        # Avoid mutable default argument
        self._metrics_early_stop = DEFAULT_METRICS_EARLY_STOP if metrics_early_stop is None else metrics_early_stop

        # Save rest
        self._activation = activation
        self._loss = loss

    def _fit_once(self,
                  samples: Iterable[float],
                  target: float,
                  norm: float) -> float:
        # Compute output
        output = self.predict(samples)

        # Compute error
        delta = self._loss(output, target)

        # Compute new weights
        self._weights = [weight + norm * delta * self._activation.derivative(output) * input_value
                         for input_value, weight
                         in zip(samples, self._weights)]

        if self._bias is not None:
            self._bias += norm * delta * self._activation.derivative(output) * 1.

        return delta, output

    def predict(self, samples: Iterable[float]) -> float:
        # Sum of element-wise input & weights  multiplication
        net = sum(value * weight
                  for value, weight
                  in zip(samples, self._weights))

        if self._bias is not None:
            net += self._bias * 1.

        return self._activation(net)

    def _default_batch_histroian(self, verbose: bool):
        if not verbose:
            return Historian()

        fmtstr = ('N{idx}/{total}',
                  'Samples: {samples}',
                  'Target: {target}',
                  'Error: {error}')
        fmtstr = ', '.join(fmtstr)

        return Historian(fmtstr)

    def _default_epoch_histroian(self, verbose: bool):
        if not verbose:
            return Historian()

        # Metrics format string
        fmtstr = (f'{name}: {{{name}}}'
                  for name, _
                  in self._metrics)
        fmtstr = ', '.join(fmtstr)
        fmtstr = f'Metrics: [{fmtstr}]'

        # Other fields
        fmtstr = ('Epoch N{idx}',
                  'Weigths: {weights}',
                  'Bias: {bias}',
                  fmtstr)
        fmtstr = ', '.join(fmtstr)

        return Historian(fmtstr)

    def fit(self,
            data_generator: DataGenerator,
            epoch_size: int,
            norm: float,
            *,
            verbose: bool = True,
            write_batch_history: bool = True,
            write_epoch_history: bool = True,
            batch_historian: Historian = None,
            epoch_historian: Historian = None) -> Tuple[List, List]:
        if batch_historian is None and write_batch_history:
            batch_historian = self._default_batch_histroian(verbose)

        if epoch_historian is None and write_epoch_history:
            epoch_historian = self._default_epoch_histroian(verbose)

        # Consume one epoch to acquire complete list of targets for metrics
        targets = [target
                   for _, target
                   in data_generator.epoch] if self._metrics else list()

        for idx, epoch in data_generator.eternity:
            outputs = list()

            for jdx, (samples, target) in epoch:
                error, output = self._fit_once(samples, target)

                if write_batch_history:
                    # Form kwargs
                    data = {
                        'idx': jdx,
                        'total': data_generator.epoch_size,
                        'samples': samples,
                        'target': target,
                        'error': error,
                    }

                    batch_historian.append(**data)

                # Save outputs for metrics
                outputs.append(output)

            # Compute every metric
            metrics = {name: metric(outputs, targets)
                       for name, metric
                       in self._metrics}

            if write_epoch_history:
                # Form kwargs once again
                data = {
                    'idx': idx,
                    'weights': self._weights,
                    'bias': self._bias,
                    **metrics,
                }

                batch_historian.append(**data)

            # Check if any early stop shoots
            try:
                for key, value in metrics.items():
                    early_stop = self._metrics_early_stop.get(key)

                    # Cant simply break since check happens in nested loop
                    if early_stop(value):
                        raise StopIteration
            except StopIteration:
                break

        # Return histories
        return (list() if batch_historian is None else batch_historian.storage,
                list() if epoch_historian is None else epoch_historian.storage)
