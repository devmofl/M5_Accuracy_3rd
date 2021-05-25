import itertools
from typing import Dict, Iterable, Iterator, Optional

import numpy as np
import torch

from pts.transform.transform import Transformation
from .common import DataEntry, Dataset


class TransformedIterableDataset(torch.utils.data.IterableDataset):
    def __init__(
        self, dataset: Dataset, is_train: bool, transform: Transformation, is_forever: bool = True,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.transform = transform
        self.is_train = is_train
        self._cur_iter: Optional[Iterator] = None
        self.is_forever = is_forever

    def _iterate_forever(self, collection: Iterable[DataEntry]) -> Iterator[DataEntry]:
        # iterate forever over the collection, the collection must be non empty
        while True:
            try:
                first = next(iter(collection))
            except StopIteration:
                raise Exception("empty dataset")
            else:
                for x in collection:
                    yield x

    def __iter__(self) -> Iterator[Dict[str, np.ndarray]]:
        if self.is_forever:
            if self._cur_iter is None:
                # only set once
                self._cur_iter = self.transform(
                    self._iterate_forever(self.dataset), is_train=self.is_train
                )
        else:
            # reset at start
            self._cur_iter = self.transform(
                self.dataset, is_train=self.is_train
            )
            
        assert self._cur_iter is not None
        while True:
            try:
                data_entry = next(self._cur_iter)
            except StopIteration:
                return
            yield {
                k: (v.astype(np.float32) if v.dtype.kind == "f" else v)
                for k, v in data_entry.items()
                if isinstance(v, np.ndarray) == True
            }

    # def __len__(self) -> int:
    #     return len(self.dataset)

class TransformedListDataset(torch.utils.data.Dataset):
    def __init__(
        self, dataset: list, is_train: bool, transform: Transformation, 
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.transform = transform
        self.is_train = is_train

    def __getitem__(self, idx):
        data_item = self.transform(
                        [self.dataset[idx]], is_train=self.is_train
                    )
        data_entry = next(data_item)

        return {
            k: (v.astype(np.float32) if v.dtype.kind == "f" else v)
            for k, v in data_entry.items()
            if isinstance(v, np.ndarray) == True
        }

    def __len__(self) -> int:
        return len(self.dataset)

    
