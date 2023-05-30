# Copyright 2020 Toyota Research Institute.  All rights reserved.

import cv2  # type: ignore[import]
import numpy as np
import numpy.typing as npt
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import optim

from sips.data import SonarBatch
from sips.networks import InlierNet, KeypointNet, KeypointResnet


class KeypointNetwithIOLoss(pl.LightningModule):
    """
    Model class encapsulating the KeypointNet and the IONet.
    """

    def __init__(
        self,
        keypoint_loss_weight: float = 1.0,
        descriptor_loss_weight: float = 2.0,
        score_loss_weight: float = 1.0,
        keypoint_net_learning_rate: float = 0.001,
        with_io: bool = True,
        use_color: bool = True,
        do_upsample: bool = True,
        do_cross: bool = True,
        descriptor_loss: bool = True,
        with_drop: bool = True,
        keypoint_net_type: str = "KeypointNet",
        opt_learn_rate: float = 0.001,
        opt_weight_decay: float = 0.0,
        sched_decay_rate: float = 0.5,
        sched_decay_frequency: int = 50,
        **kwargs,
    ) -> None:

        super().__init__()

        self.keypoint_loss_weight = keypoint_loss_weight
        self.descriptor_loss_weight = descriptor_loss_weight
        self.score_loss_weight = score_loss_weight
        self.keypoint_net_learning_rate = keypoint_net_learning_rate

        self.cell = 8  # Size of each output cell. Keep this fixed.
        self.border_remove = 4  # Remove points this close to the border.
        self.top_k2 = 300
        self.relax_field = 4

        self.use_color = use_color
        self.descriptor_loss = descriptor_loss

        # Set optimizer and scheduler parameters
        self.opt_learn_rate = opt_learn_rate
        self.opt_weight_decay = opt_weight_decay
        self.sched_decay_rate = sched_decay_rate
        self.sched_decay_frequency = sched_decay_frequency

        # Initialize KeypointNet
        self.keypoint_net: KeypointNet | KeypointResnet
        if keypoint_net_type == "KeypointNet":
            self.keypoint_net = KeypointNet(
                use_color=use_color,
                do_upsample=do_upsample,
                with_drop=with_drop,
                do_cross=do_cross,
            )
        elif keypoint_net_type == "KeypointResnet":
            self.keypoint_net = KeypointResnet(with_drop=with_drop)
        else:
            msg = f"Keypoint net type not supported {keypoint_net_type}"
            raise NotImplementedError(msg)

        # Initialize IO-Net
        self.with_io = with_io
        self.io_net = InlierNet(blocks=4) if self.with_io else None

        # Other useful things to track
        self.vis: dict[str, npt.NDArray[np.float64]] = {}

    def configure_optimizers(self) -> optim.Optimizer:
        return optim.Adam(
            self.parameters(),
            lr=self.opt_learn_rate,
            weight_decay=self.opt_weight_decay,
        )

    def forward(self, batch: SonarBatch):
        raise NotImplementedError

    def training_step(self, batch: SonarBatch, batch_idx: int):
        raise NotImplementedError

    def validation_step(self, batch: SonarBatch, batch_idx: int):
        raise NotImplementedError

    def test_step(self, batch: SonarBatch, batch_idx: int):
        raise NotImplementedError
