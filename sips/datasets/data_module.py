import json

import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, PILToTensor, Resize

from sips.configs.base_config import _DatasetsConfig
from sips.data import CameraParams, CameraPose, SonarBatch, SonarDatumPair
from sips.datasets.dataset import DummySonarDataSet, SonarDataset


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

        path = "data/filtered/freeRoaming15SonarFeSteg-0805"
        ts1, ts2 = 1683542806144695444, 1683542806397509699
        cam_param = CameraParams(1, 10, 130, 20)

        trsf = Compose([PILToTensor(), Resize(size=(480, 480), antialias=True)])
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

        sonar_pair = SonarDatumPair((img1, pose1, cam_param), (img2, pose2, cam_param))

        self.data_train: SonarDataset = DummySonarDataSet(n=64, sonar_pair=sonar_pair)
        self.data_val: SonarDataset = DummySonarDataSet(n=2, sonar_pair=sonar_pair)

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

    def train_dataloader(self) -> DataLoader[SonarDatumPair]:
        return DataLoader(
            self.data_train,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            collate_fn=SonarBatch,
        )

    def val_dataloader(self) -> DataLoader[SonarDatumPair]:
        return DataLoader(
            self.data_val,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            collate_fn=SonarBatch,
        )
