import json
from pathlib import Path
from warnings import warn

import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose, PILToTensor, Resize

from sips.configs.base_config import _DatasetsConfig
from sips.data import CameraParams, CameraPose, SonarBatch, SonarDatumPair
from sips.data_extraction.init_preprocessing import call_preprocessing_steps
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
        self.tuples: SonarDataset

    def prepare_data(self) -> None:
        """
        Run all preprocessing steps for the provided rosbags and configuration
        if not already done before. Combine the poses and tuple timestamps from
        all rosbags.

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
        rosbags = self.config.rosbags
        poses: dict[int, dict[str, dict[str, float] | int | str]] = {}
        tuples: list[tuple[int, int]] = []
        for this_rosbag in rosbags:
            print(100 * "=")
            print(f"Start preprocessing {this_rosbag}")
            source_dir = call_preprocessing_steps(self.config, this_rosbag)
            if source_dir == None:
                continue
            source_dir = Path(source_dir)  # type:ignore[arg-type]
            try:
                with open(source_dir / "unfiltered_pose_data.json") as f:
                    pose_data = json.load(f)
            except IOError:
                warn(f"unfiltered_pose_data.json file missing of {this_rosbag}")
                continue
            except json.JSONDecodeError:
                warn(f"unfiltered_pose_data.json file empty of {this_rosbag}")
                continue
            try:
                with open(source_dir / "tuple_stamps.json") as f:
                    these_tuples = json.load(f)
            except IOError:
                warn(f"tuple_stamps.json file missing of {this_rosbag}")
                continue
            except json.JSONDecodeError:
                warn(f"tuple_stamps.json file empty of {this_rosbag}")
                continue
            config_num = int(source_dir.name)
            for this_pose in pose_data:
                poses[int(this_pose["timestamp"])] = {
                    "point_position": this_pose["point_position"],
                    "quaterion_orientation": this_pose["quaterion_orientation"],
                    "rosbag": this_rosbag.split(".")[0],
                    "config_num": config_num,
                }
            tuples += these_tuples
        with open("data/poses.json", "w") as f:
            json.dump(poses, f, indent=2)
        with open("data/tuples.json", "w") as f:
            json.dump(tuples, f, indent=2)

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
        dataset = SonarDataset(self.config)
        train_set, val_set = random_split(
            dataset, [self.config.train_ratio, self.config.val_ratio]
        )

        self.data_train = SonarDataset(self.config, train_set)
        self.data_val = SonarDataset(self.config, val_set)

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
