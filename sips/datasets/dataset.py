from warnings import warn

from torch.utils.data import Dataset

from scripts.examples._sonar_data import get_random_datum_tuple
from sips.dataclasses import SonarDatumTuple


class SonarDataset(Dataset[SonarDatumTuple]):
    # TODO: Implement

    def __init__(self) -> None:
        super().__init__()

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index) -> SonarDatumTuple:
        # img = Image.open(self.paths[index]).convert("RGB")
        a = ...
        b = ...
        return SonarDatumTuple(a, b)


class DummySonarDataSet(SonarDataset):
    def __init__(self, n: int = 100) -> None:
        warn("WARNING: You are using a dummy dataset!")
        super().__init__()
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index) -> SonarDatumTuple:
        return get_random_datum_tuple()
