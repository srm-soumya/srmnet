"""
Data comes as batches
"""
import numpy as np
from typing import Iterator, NamedTuple

from srmnet.tensor import Tensor

Batch = NamedTuple("Batch", [("inputs", Tensor), ("targets", Tensor)])

class DataIterator:
    def __call__(self, inputs: Tensor, targets: Tensor) -> Iterator:
        raise NotImplementedError

class BatchIterator(DataIterator):
    def __init__(self, bs: int=32, shuffle: bool=True) -> None:
        self.bs = bs
        self.shuffle = shuffle

    def __call__(self, inputs: Tensor, targets: Tensor) -> Iterator:
        starts = np.arange(0, len(inputs), self.bs)
        if self.shuffle:
            np.random.shuffle(starts)

        for start in starts:
            end = start + self.bs
            batch_inputs =  inputs[start:end]
            batch_targets = targets[start:end]
            yield Batch(batch_inputs, batch_targets)
