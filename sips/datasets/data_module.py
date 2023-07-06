import json

import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, PILToTensor, Resize

from sips.configs.base_config import _DatasetsConfig
from sips.data import CameraParams, CameraPose, SonarBatch, SonarDatumPair
from sips.datasets.dataset import DummySonarDataset, SonarDataset, _SonarDatasetBase

# ==========================================================================
# Base


def _make_dataloader(
    dataset: _SonarDatasetBase, config: _DatasetsConfig
) -> DataLoader[SonarDatumPair]:
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        collate_fn=SonarBatch,
    )


class _SonarDataModuleBase(pl.LightningDataModule):
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
        raise NotImplementedError

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
        raise NotImplementedError


# ==========================================================================
# Data Modules


class SonarDataModule(_SonarDataModuleBase):
    def __init__(self, config: _DatasetsConfig) -> None:
        super().__init__(config)

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
        self.data_train: _SonarDatasetBase = DummySonarDataset(n=100)
        self.data_val: _SonarDatasetBase = DummySonarDataset(n=10)

    def train_dataloader(self) -> DataLoader[SonarDatumPair]:
        return _make_dataloader(self.data_train, self.config)

    def val_dataloader(self) -> DataLoader[SonarDatumPair]:
        return _make_dataloader(self.data_val, self.config)


class DummySonarDataModule(_SonarDataModuleBase):
    def __init__(self, config: _DatasetsConfig) -> None:
        super().__init__(config)

    def prepare_data(self) -> None:
        """
        Download, split, etc... data

        Notes
        -----
        Only called on 1 GPU/TPU in distributed

        """
        path = "data/filtered/freeRoaming15SonarFeSteg-0805"
        ts1, ts2 = 1683542806144695444, 1683542806397509699
        cam_param = CameraParams(1, 10, 130, 20)

        trsf = Compose(
            [PILToTensor(), Resize(size=self.config.image_shape, antialias=True)]  # type: ignore
        )
        img1: torch.Tensor = trsf(Image.open(f"{path}/images/sonar_{ts1}.png"))  # type: ignore
        img2: torch.Tensor = trsf(Image.open(f"{path}/images/sonar_{ts2}.png"))  # type: ignore

        with open(f"{path}/pose_data.json") as f:
            pose_data = json.load(f)
        pose1 = pose2 = None
        for pose_datum in pose_data:
            ts = pose_datum["timestamp"]
            position = [pose_datum["point_position"][ax] for ax in "xyz"]
            rotation = [pose_datum["quaterion_orientation"][ax] for ax in "xyzw"]
            if ts == ts1:
                pose1 = CameraPose(position, rotation)
            elif ts == ts2:
                pose2 = CameraPose(position, rotation)
        assert pose1 is not None and pose2 is not None

        self.sonar_pair = SonarDatumPair(
            (img1, pose1, cam_param), (img2, pose2, cam_param)
        )

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
        self.data_train = DummySonarDataset(
            n=8 * self.config.batch_size, sonar_pair=self.sonar_pair
        )
        self.data_val = DummySonarDataset(
            n=self.config.batch_size, sonar_pair=self.sonar_pair
        )

    def train_dataloader(self) -> DataLoader[SonarDatumPair]:
        return _make_dataloader(self.data_train, self.config)

    def val_dataloader(self) -> DataLoader[SonarDatumPair]:
        return _make_dataloader(self.data_val, self.config)
