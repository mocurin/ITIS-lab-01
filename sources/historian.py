"""Data-writer & logger class"""
import logging
import enum

import matplotlib.pyplot as plt

from typing import Callable, Dict, List

from . import models


class LoggingVerbosity(enum.IntEnum):
    SILENCE = 0
    ETERNITY = 1
    EPOCH = 2
    BATCH = 3
    SAMPLE = 4


class WriteVerbosity(enum.IntEnum):
    SILENCE = 0
    ETERNITY = 1
    EPOCH = 2
    BATCH = 3
    SAMPLE = 4


class FitEvents(enum.IntEnum):
    ETERNITY = 1
    EPOCH = 2
    BATCH = 3
    SAMPLE = 4


class Historian:
    def __init__(self,
                 logging_verbosity: LoggingVerbosity = LoggingVerbosity.EPOCH,
                 write_verbosity: WriteVerbosity = WriteVerbosity.EPOCH,
                 describers: Dict[FitEvents, Callable] = None):
        self._logging_verbosity = logging_verbosity
        self._write_verbosity = write_verbosity

        # Avoid mutable default arguments
        if describers is None:
            describers = dict()

        # Do not check wether all describers are implemented
        self._describers = describers
        self._storage = {
            event: list()
            for event
            in list(FitEvents)
            if not event > write_verbosity
        }

    def store(self, event: FitEvents, model: 'models.Model', *args):
        if not event > self._logging_verbosity:
            # Describers could be missing
            describer = self._describers.get(event)

            if describer:
                describer(model, *args)

        if not event > self._write_verbosity:
            # Storages could not be missing
            self._storage[event].append([*model.describe(), *args])


class NeuronHistorian(Historian):
    def __init__(self,
                 logging_verbosity: LoggingVerbosity.EPOCH,
                 write_verbosity: WriteVerbosity = WriteVerbosity.EPOCH):
        # Make methods accessible through storage
        describers = {
            FitEvents.BATCH: NeuronHistorian.describe_batch,
            FitEvents.EPOCH: NeuronHistorian.describe_epoch,
            FitEvents.ETERNITY: NeuronHistorian.describe_eternity
        }
        super().__init__(
            logging_verbosity,
            write_verbosity,
            describers
        )

    @staticmethod
    def _describe_model(model: 'models.Neuron'):
        # Prepare weights
        weights = [
            f"{weight:.4f}"
            for weight
            in model._weights
        ]

        return [
            f"W=[{', '.join(weights)}]"
        ]

    @staticmethod
    def describe_epoch(model: 'models.Neuron',
                       epoch_idx: int,
                       outputs: List[int],
                       metrics: Dict[str, int]):
        metric_repr = [
            f"{metric}={result:.4f}"
            for metric, result
            in metrics.items()
        ]

        # VScode is not happy about f-string being carried over to a new line
        string_repr = [
            f"E{epoch_idx:3}",
            *NeuronHistorian._describe_model(model),
            f"Y={outputs}",
            f"{' '.join(metric_repr)}"
        ]

        logging.info(' '.join(string_repr))

    @staticmethod
    def describe_batch(model: 'models.Neuron',
                       batch_idx: int,
                       samples: List[int],
                       output: int,
                       error: int):
        string_repr = [
            f"B{batch_idx:3}",
            *NeuronHistorian._describe_model(model),
            f"x={samples}",
            f"y={output}",
            f"d={error}"
        ]

        logging.info(' '.join(string_repr))

    @staticmethod
    def describe_eternity(model: 'models.Neuron',
                          historian: 'NeuronHistorian',
                          stop_event: 'models.FitStopEvents',
                          prefix: str = 'neuron'):
        if stop_event is models.FitStopEvents.METRIC_STOP and historian._write_verbosity >= WriteVerbosity.EPOCH:
            # Acquire epoch data
            rows = historian._storage[WriteVerbosity.EPOCH]
            hamming = [(row[1], row[-1]['E']) for row in rows]

            # Transpose to create 2 lists
            epoch_idx, hamming = map(list, zip(*hamming))

            # Plot fitting result
            plt.title(prefix)
            plt.grid()
            plt.xlim(0, max(10, max(epoch_idx)))
            plt.ylim(0, max(14, max(hamming)))
            plt.plot(epoch_idx, hamming)
            plt.savefig(f"{prefix}_eternity.png")
            plt.show()
