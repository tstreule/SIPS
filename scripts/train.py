"""
Script for SIPS training.

"""
from sips.utils.dotenv import load_dotenv

load_dotenv()


from dataclasses import asdict
from typing import Optional

import torch._dynamo
import typer
from pytorch_lightning import Trainer, seed_everything
from typing_extensions import Annotated

from sips.configs import parse_train_file
from sips.datasets import SonarDataModule
from sips.models import KeypointNetwithIOLoss
from sips.utils.logging import setup_wandb_logger

app = typer.Typer()


def _torch_compile(model: KeypointNetwithIOLoss, **kwargs) -> KeypointNetwithIOLoss:
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
def main(config_file: Annotated[Optional[str], typer.Option("--config")] = None):
    # Read configuration
    config_file = config_file or "sips/configs/v0_dummy.yaml"  # TODO: Delete this line
    config = parse_train_file(config_file)

    # Initialize model and data module
    model = KeypointNetwithIOLoss(**asdict(config.model))
    model = _torch_compile(model)
    dm = SonarDataModule(config.datasets)

    # Initialize trainer and make everything reproducible
    # c.f. https://lightning.ai/docs/pytorch/stable/common/trainer.html#reproducibility
    seed_everything(config.arch.seed, workers=True)
    trainer = Trainer(
        deterministic=True,
        # Strategy
        strategy=config.arch.strategy,
        accelerator=config.arch.accelerator,
        devices=config.arch.devices,
        # Training args
        max_epochs=config.arch.max_epochs,
        callbacks=None,
        logger=setup_wandb_logger(config),
        fast_dev_run=config.arch.fast_dev_run,
        log_every_n_steps=config.arch.log_every_n_steps,
    )

    # Train model
    trainer.fit(model, datamodule=dm, ckpt_path=None)

    # Test (only right before publishing your paper or pushing to production!)
    # trainer.test(datamodule=dm, ckpt_path="best")


if __name__ == "__main__":
    app()
