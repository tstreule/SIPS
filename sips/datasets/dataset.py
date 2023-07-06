from warnings import warn

from torch.utils.data import Dataset

from scripts.examples._sonar_data import get_random_datum_pair
from sips.data import SonarDatumPair


class _SonarDatasetBase(Dataset[SonarDatumPair]):
    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index) -> SonarDatumPair:
        raise NotImplementedError


class SonarDataset(_SonarDatasetBase):
    # TODO: Implement

    def __init__(self) -> None:
        super().__init__()

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index) -> SonarDatumPair:
        # img = Image.open(self.paths[index]).convert("RGB")
        a = ...
        b = ...
        return SonarDatumPair(a, b)  # type: ignore[arg-type]


class DummySonarDataset(_SonarDatasetBase):
    def __init__(self, n: int = 100, sonar_pair: SonarDatumPair | None = None) -> None:
        warn("WARNING: You are using a dummy dataset!")
        super().__init__()

        self.n = n
        self.sonar_pair = sonar_pair

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index) -> SonarDatumPair:
        return self.sonar_pair or get_random_datum_pair()
