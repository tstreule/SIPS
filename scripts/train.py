"""
Script for SIPS training.

"""
from sips.utils.dotenv import load_dotenv

load_dotenv()


from dataclasses import asdict
from typing import Optional

import typer
from pytorch_lightning import Trainer, seed_everything
from typing_extensions import Annotated

from sips.configs import parse_train_file
from sips.datasets import SonarDataModule
from sips.models import KeypointNetwithIOLoss
from sips.utils.logging import setup_wandb_logger

app = typer.Typer()


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
        logger=setup_wandb_logger(config.wandb),
        fast_dev_run=config.arch.fast_dev_run,
    )

    # Train model
    trainer.fit(model, datamodule=dm, ckpt_path=None)

    # Test (only right before publishing your paper or pushing to production!)
    # trainer.test(datamodule=dm, ckpt_path="best")


if __name__ == "__main__":
    app()
