import json
from json import JSONDecodeError

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from scripts import preprocess_data
from sips.configs.base_config import _DatasetsConfig
from sips.data import SonarBatch, SonarDatumPair
from sips.data_extraction.preprocessing import call_preprocessing_steps
from sips.datasets.dataset import DummySonarDataSet, SonarDataset


class SonarDataModule(pl.LightningDataModule):
    def __init__(self, config: _DatasetsConfig) -> None:
        super().__init__()
        # If set to True will call prepare_data() on LOCAL_RANK=0 for every node.
        # If set to False will only call from NODE_RANK=0, LOCAL_RANK=0.
        self.prepare_data_per_node = False

        self.config = config
        self.data_tuples = []

    def prepare_data(self) -> None:
        """
        Download, split, etc... data

        Notes
        -----
        Only called on 1 GPU/TPU in distributed

        """
        ...
        # TODO: make it so that all data preprocessing can be executed from here
        # already looked at bags are skipped, to do this only use few lines of code in here
        # i.e. introduce scripts that can be called from here.

        # MS: combine all the datapoints of valid bags (i.e. those that have enough data)
        # randomize order and store in folder according to config of this module
        rosbags = self.config.rosbags
        for this_rosbag in rosbags:
            source_dir = call_preprocessing_steps(self.config, this_rosbag)
            try:
                with open(source_dir / "tuple_stamps.json") as f:
                    tuple_stamps = json.load(f)
            except IOError:
                warn(f"tuple_stamps.json file missing of {this_rosbag}")
                continue
            except JSONDecodeError:
                warn(f"tuple_stamps.json file empty of {this_rosbag}")
                continue
            # for this_tuple_stamp in tuple_stamps:

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
        # MS: include test loader as well
        self.data_train: SonarDataset = DummySonarDataSet(n=100)
        self.data_val: SonarDataset = DummySonarDataSet(n=10)

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
