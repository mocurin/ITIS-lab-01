"""Common data generator package"""
from typing import Generator, Sequence, Tuple
from itertools import compress, cycle, islice, product
from more_itertools import distinct_permutations


# Type hinting
RawDataGenerator = Generator[Tuple[Sequence[float], float], None, None]


class DataGenerator:
    def __init__(self, generator: RawDataGenerator, epoch_size: int, stop_epoch: int = None):
        self._generator = generator
        self._epoch_size = epoch_size
        self._stop_epoch = stop_epoch

    @property
    def epoch(self):
        return islice(self._generator, self._epoch_size)

    @property
    def eternity(self):
        return (self.epoch for _ in range(self._stop_epoch))

    @property
    def epoch_size(self):
        return self._epoch_size


def boolean_str_to_seq(boolean: str) -> Sequence[int]:
    return [int(sym) for sym in boolean]


def boolean_generator(boolean: Sequence[int]) -> RawDataGenerator:
    # Check whether `len(boolean)` is a power of two
    if not (size := len(boolean)) or (size & (size - 1)):
        raise ValueError(f"Given sequence ({boolean}) length does not equal power of two: {size}")

    # Repeat values from (0, 1) `len(boolean)` times in vectors of `log2(len(boolean))` elements
    rows = product((0, 1), repeat=len(boolean).bit_length() - 1)

    # Return indefinite generator of pairs
    return cycle(zip(rows, boolean))


def subset(generator: Generator, mask: Sequence) -> Generator:
    chunked = islice(generator, len(mask))
    return cycle(compress(chunked, mask))


def boolean_mask_generator(length: int):
    return (distinct_permutations([0] * idx + [1] * (length - idx))
            for idx
            in range(length))
