from dataclasses import dataclass

import torch
from torch.utils.data import IterableDataset

from sips.dataclasses import SonarDatumTuple


class SonarDataset(IterableDataset[SonarDatumTuple]):
    # TODO: Implement

    def __init__(self) -> None:
        super().__init__()

    def __getitem__(self, index) -> SonarDatumTuple:
        # img = Image.open(self.paths[index]).convert("RGB")
        a = ...
        b = ...
        return SonarDatumTuple(a, b)
