from warnings import warn

from torch.utils.data import Dataset

from scripts.examples._sonar_data import get_random_datum_pair
from sips.data import SonarDatumPair


class SonarDataset(Dataset[SonarDatumPair]):
    # TODO: Implement

    def __init__(self) -> None:
        super().__init__()

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index) -> SonarDatumPair:
        # img = Image.open(self.paths[index]).convert("RGB")
        a = ...
        b = ...
        return SonarDatumPair(a, b)


class DummySonarDataSet(SonarDataset):
    def __init__(self, n: int = 100) -> None:
        warn("WARNING: You are using a dummy dataset!")
        super().__init__()
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index) -> SonarDatumPair:
        return get_random_datum_pair()
