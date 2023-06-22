import json
from json import JSONDecodeError
from pathlib import Path
from warnings import warn

import pytorch_lightning as pl
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, random_split

from scripts import preprocess_data
from sips.configs.base_config import _DatasetsConfig
from sips.data import CameraParams, CameraPose, SonarBatch, SonarDatum, SonarDatumPair
from sips.data_extraction.preprocessing import call_preprocessing_steps
from sips.datasets.dataset import DummySonarDataSet, SonarDataset


class SonarDataModule(pl.LightningDataModule):
    def __init__(self, config: _DatasetsConfig) -> None:
        super().__init__()
        # If set to True will call prepare_data() on LOCAL_RANK=0 for every node.
        # If set to False will only call from NODE_RANK=0, LOCAL_RANK=0.
        self.prepare_data_per_node = False

        self.config = config
        self.tuples: SonarDataset

    def prepare_data(self) -> None:
        """
        Download, split, etc... data

        Notes
        -----
        Only called on 1 GPU/TPU in distributed

        """

        # MS: combine all the datapoints of valid bags (i.e. those that have enough data)
        # randomize order and store in folder according to config of this module
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
            except JSONDecodeError:
                warn(f"unfiltered_pose_data.json file empty of {this_rosbag}")
                continue
            try:
                with open(source_dir / "tuple_stamps.json") as f:
                    these_tuples = json.load(f)
            except IOError:
                warn(f"tuple_stamps.json file missing of {this_rosbag}")
                continue
            except JSONDecodeError:
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

        # MS: include test loader as well
        self.data_train = SonarDataset(self.config, train_set)
        self.data_val = SonarDataset(self.config, val_set)

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
