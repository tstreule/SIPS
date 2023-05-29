import pytorch_lightning as pl
from torch.utils.data import DataLoader

from sips.configs.base_config import _DatasetsConfig
from sips.dataclasses import SonarDatumTuple
from sips.datasets.dataset import DummySonarDataSet, SonarDataset


def sonar_collate_fn(batch: list[SonarDatumTuple]) -> list[SonarDatumTuple]:
    assert isinstance(batch[0], SonarDatumTuple)
    # TODO: This is is not yet nice
    return batch


class SonarDataModule(pl.LightningDataModule):
    def __init__(self, config: _DatasetsConfig) -> None:
        super().__init__()
        # If set to True will call prepare_data() on LOCAL_RANK=0 for every node.
        # If set to False will only call from NODE_RANK=0, LOCAL_RANK=0.
        self.prepare_data_per_node = False

        self.config = config

    def prepare_data(self) -> None:
        """
        Download, split, etc... data

        Notes
        -----
        Only called on 1 GPU/TPU in distributed

        """
        ...

    def setup(self, stage: str) -> None:
        """
        Make assignments here (val/train/test split).

        Parameters
        ----------
        stage : str
            Either 'train', 'validate', 'test', or 'predict'.

        Notes
        -----
        Called on every process in DDP

        """
        ...
        self.data_train: SonarDataset = DummySonarDataSet(n=100)
        self.data_val: SonarDataset = DummySonarDataSet(n=10)

    def train_dataloader(self) -> DataLoader[SonarDatumTuple]:
        return DataLoader(
            self.data_train,
            batch_size=self.config.batch_size,
            collate_fn=sonar_collate_fn,
        )

    def val_dataloader(self) -> DataLoader[SonarDatumTuple]:
        return DataLoader(
            self.data_val,
            batch_size=self.config.batch_size,
            collate_fn=sonar_collate_fn,
        )
