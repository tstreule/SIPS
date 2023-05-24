import pytorch_lightning as pl
from torch.utils.data import DataLoader

from sips.configs.base_config import _DatasetsConfig
from sips.dataclasses import SonarDatumTuple
from sips.datasets.dataset import SonarDataset


class SonarDataModule(pl.LightningDataModule):
    def __init__(self, config: _DatasetsConfig) -> None:
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
        self.data_train: SonarDataset = ...  # type: ignore[assignment]
        self.data_val: SonarDataset = ...  # type: ignore[assignment]

    def train_dataloader(self) -> DataLoader[SonarDatumTuple]:
        return DataLoader(self.data_train, batch_size=self.config.train.batch_size)

    def val_dataloader(self) -> DataLoader[SonarDatumTuple]:
        return DataLoader(self.data_val, batch_size=self.config.train.batch_size)
