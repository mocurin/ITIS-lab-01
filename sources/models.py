"""Models package"""
import abc
import enum

from itertools import chain
from typing import Any, Callable, Dict, Iterable, List

from .activations import identity, Activation
from .generators import DataGenerator
from .initializers import zeros, Initializer
from .historian import NeuronHistorian, LoggingVerbosity, WriteVerbosity, FitEvents
from .losses import difference, Loss
from .metrics import hamming, Metric
from .misc import subclasshook_helper


_REQUIRED = (
    '__call__',
    'fit',
    'predict',
    'describe'
)


class IModel(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass) -> bool:
        return subclasshook_helper(_REQUIRED)

    @abc.abstractmethod
    def fit(self, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def describe(selg):
        raise NotImplementedError


# Type hinting alias
Model = IModel
MetricEarlyStop = Callable[[Any], bool]


DEFAULT_METRICS = {
    'E': hamming
}

DEFAULT_METRICS_EARLY_STOP = {
    'E': lambda x: x == 0
}


class FitStopEvents(enum.IntEnum):
    EPOCH_STOP = 0
    METRIC_STOP = 1
    STALE_STOP = 2


class Neuron(IModel):
    historian = NeuronHistorian

    def __init__(self,
                 inputs: int,
                 *,
                 use_bias: bool = True,
                 activation: Activation = identity,
                 weights_initializer: Initializer = zeros(),  # Intentional mutable default argument
                 bias_initializer: Initializer = zeros(),  # Intentional mutable default argument
                 loss: Loss = difference,
                 metrics: Dict[str, Metric] = None,
                 metrics_early_stop: Dict[str, MetricEarlyStop] = None):

        # Generate weights by taking `len(inputs)` times from `weights_initializer`
        self._weights = [weight
                         for _, weight
                         in zip(range(inputs), weights_initializer)]

        self._use_bias = use_bias

        # Append bias weight in front
        if use_bias:
            self._weights = [next(bias_initializer), *self._weights]

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
        # Compute output, preserve net value
        net = self.predict(samples, activate=False)
        output = self._activation(net)

        # Compute error
        delta = self._loss(output, target)

        # Append fake sample in front of given if bias is used
        if self._use_bias:
            samples = chain([1.], samples)

        # Compute new weights
        self._weights = [
            weight + norm * delta * self._activation.derivative(net) * input_value
            for input_value, weight
            in zip(samples, self._weights)
        ]

        return delta, output

    def predict(self, samples: Iterable[float], activate=True) -> float:
        # Append fake sample in front of given if bias is used
        if self._use_bias:
            samples = chain([1.], samples)

        # Sum of element-wise input & weights  multiplication
        net = sum(
            value * weight
            for value, weight
            in zip(samples, self._weights)
        )

        # Apply activation based in `activate` boolean
        return self._activation(net) if activate else net

    def fit(self,
            train_data_generator: DataGenerator,
            validation_data_generator: DataGenerator,
            norm: float, delta: float, *,
            logging_verbosity: LoggingVerbosity = LoggingVerbosity.EPOCH,
            write_verbosity: WriteVerbosity = WriteVerbosity.EPOCH,
            plot_prefix: str = 'neuron') -> List:
        # Create logger & history writer
        historian = self.historian(logging_verbosity, write_verbosity)

        # Predict stop event
        stop_event = FitStopEvents.EPOCH_STOP

        for idx, epoch in enumerate(train_data_generator.eternity):
            # Compute score over epoch of validation data
            outputs = [
                (self.predict(sample), target)
                for sample, target
                in validation_data_generator.epoch
            ]

            # Transpose list of pair to a pair of lists
            outputs, targets = map(list, zip(*outputs))

            # Compute every metric
            metrics = {
                name: metric(outputs, targets)
                for name, metric
                in self._metrics.items()
            }

            # Write epoch history
            historian.store(
                FitEvents.EPOCH,
                self,
                idx,
                outputs,
                metrics
            )

            # Check if any metric early stop shoots
            for name, checker in self._metrics_early_stop.items():
                metric = metrics[name]

                if checker(metric):
                    # Mark as stopped from metric
                    stop_event = FitStopEvents.METRIC_STOP
                    break

            # Break out outer cycle after metric stop
            if stop_event == FitStopEvents.METRIC_STOP:
                break

            # Remember weights for stale stop check
            before_weights = [weight for weight in self._weights]

            # Fit on every sample from epoch
            for jdx, (sample, target) in enumerate(epoch):
                error, output = self._fit_once(sample, target, norm)

                # Write batch history
                historian.store(
                    FitEvents.BATCH,
                    self,
                    jdx,
                    sample,
                    output,
                    error
                )

            # Check if any weights changed
            total_change = sum(
                abs(bw - aw)
                for bw, aw
                in zip(before_weights, self._weights)
            )

            # Mark as stopped from stale weights
            if total_change < delta:
                stop_event = FitStopEvents.STALE_STOP
                break

        # Write eternity history
        historian.store(
            FitEvents.ETERNITY,
            self,
            historian,
            stop_event,
            plot_prefix
        )

        # Eternity end
        return historian, stop_event

    def describe(self):
        # Return model associated data
        return [
            self._weights,
        ]
