"""
Script for SIPS training.

"""
from sips.utils.dotenv import load_dotenv

load_dotenv()


import json
from platform import python_version
from typing import Optional

import torch._dynamo
from packaging import version
from pytorch_lightning import Trainer, seed_everything
from typing_extensions import Annotated

from sips.callbacks import get_callbacks
from sips.configs import Config, parse_train_file
from sips.datasets import DummySonarDataModule, SonarDataModule
from sips.models import KeypointNetwithIOLoss
from sips.utils.logging import setup_wandb_logger

try:
    import typer
except ModuleNotFoundError:
    from sips.utils import typer  # type: ignore[no-redef]


app = typer.Typer()


def _set_debug(debug: bool) -> None:
    if not debug:
        return

    import torch
    from torch import autograd

    autograd.set_detect_anomaly(True)  # type: ignore[attr-defined]
    torch.set_printoptions(precision=2, threshold=100, edgeitems=2, sci_mode=False)
    torch._dynamo.config.verbose = True  # type: ignore[attr-defined]


def _set_accelerator(acc: str) -> str:
    # Pytorch Lightning would automatically set "mps" if accelerator is "auto" and mps
    #  is available.  However, since (1) `torch.compile()` cannot handle MPS yet and
    #  (2) some operators are not yet natively implemented in MPS, we unset it.
    if acc == "mps" or (acc == "auto" and torch.has_mps):
        acc = "cpu"

    return acc


def _set_config(config_str: Optional[str]) -> Config:
    # No config
    if not config_str:
        return Config()
    # Config path
    if config_str.rsplit(".")[-1] in ("yml", "yaml"):
        return parse_train_file(config_str)
    # Config dict
    config_dict = json.loads(
        config_str.replace("'", '"')
        .replace('"true"', "true")
        .replace('"false"', "false")
        .replace('"null"', "null")
    )
    return Config(**config_dict)


def _torch_compile(
    model: KeypointNetwithIOLoss, accelerator: str, **kwargs
) -> KeypointNetwithIOLoss:
    if version.parse(python_version()) >= version.parse("3.11"):
        # Python 3.11+ not yet supported for torch.compile
        return model

    if accelerator == "mps" or (accelerator == "auto" and torch.has_mps):
        # `torch.compile()` cannot handle MPS yet
        return model

    # NOTE: Ideally we would call `model = torch.compile(model)` and add a
    #  `@torch.compile(dynamic=True)` decorator to the
    #  `sips.utils.match_keypoints_2d_batch` function.
    #  However, due to a PyTorch issue this is currently not possible
    #  (see https://github.com/pytorch/pytorch/issues/99774).
    #  Therefore, we currently just compile the two models separately.
    torch._dynamo.reset()
    model.keypoint_net = torch.compile(model.keypoint_net, **kwargs)  # type: ignore[assignment]
    model.io_net = torch.compile(model.io_net, **kwargs)  # type: ignore[assignment]
    return model


@app.command()
def train(
    config_str: Annotated[
        Optional[str],
        typer.Option("--config", help="Config dict (as str) or path to config file"),
    ] = None,
    dummy: Annotated[
        bool, typer.Option("--dummy", help="If True, uses dummy data")
    ] = False,
):
    # Handle configuration
    config = _set_config(config_str)
    if not config.datasets.rosbags:
        config.datasets.rosbags = [
            "agisoft8.bag",
            "agisoft9.bag",
            "boatpolice-001.bag",
            "coast.bag",
            "freeRoaming-001.bag",
            "freeRoaming15SonarFeSteg-0805.bag",
            "freeRoaming15SonarFels2-0805.bag",
            "freeRoaming2.bag",
            "freeRoaming45deg-0805.bag",
            "riverSpeed.bag",
            "riverSpeedFar.bag",
            "smoothed3.bag",
            "smoothed_small_medium_yaw-002.bag",
            "tiefenbrunnen_agisoft3.bag",
            "tiefenbrunnen_agisoft4.bag",
        ]
    _set_debug(config.debug)
    config.arch.accelerator = _set_accelerator(config.arch.accelerator)

    # Initialize model and data module
    model = KeypointNetwithIOLoss.from_config(config.model)
    model = _torch_compile(model, config.arch.accelerator)
    dm: DummySonarDataModule | SonarDataModule
    if dummy:
        dm = DummySonarDataModule(config.datasets)
    else:
        dm = SonarDataModule(config.datasets)

    # Set up Wandb logger
    wandb_logger = setup_wandb_logger(config)
    if wandb_logger:
        wandb_logger.watch(model, log="all")

    # Initialize callbacks
    callbacks = get_callbacks(
        monitors=[("val_repeatability", "max"), ("val_matching_score", "max")],
        config=config,
    )

    # Initialize trainer and make everything reproducible
    # c.f. https://lightning.ai/docs/pytorch/stable/common/trainer.html#reproducibility
    seed_everything(config.arch.seed, workers=True)
    trainer = Trainer(
        deterministic="warn",
        # Strategy
        strategy=config.arch.strategy,
        accelerator=config.arch.accelerator,
        devices=config.arch.devices,
        precision=config.arch.precision,  # type: ignore[arg-type]
        # Training args
        default_root_dir=config.model.checkpoint_path,
        enable_checkpointing=config.model.save_checkpoint,
        max_epochs=config.arch.max_epochs,
        callbacks=callbacks,
        check_val_every_n_epoch=None,  # must be None since 'callbacks' is implemented
        logger=wandb_logger,
        log_every_n_steps=config.arch.log_every_n_steps,
        # Debugging
        fast_dev_run=config.debug,
        num_sanity_val_steps=None,
    )

    # Train model
    trainer.fit(model, datamodule=dm)

    # After training
    if wandb_logger:
        wandb_logger.experiment.unwatch(model)

    # Test (only right before publishing your paper or pushing to production!)
    # trainer.test(datamodule=dm, ckpt_path="best")


if __name__ == "__main__":
    app()
