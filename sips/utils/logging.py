from dataclasses import asdict
from typing import Literal

from pytorch_lightning.loggers.wandb import WandbLogger

from sips.configs.base_config import Config


def setup_wandb_logger(config: Config) -> Literal[False] | WandbLogger:
    wandb = config.wandb

    # Dry run / no logging
    if wandb.dry_run:
        return False

    # Init logger
    wandb_logger = WandbLogger(
        name=wandb.name,
        save_dir=wandb.save_dir,
        version=wandb.version,
        project=wandb.project,
        entity=wandb.entity,
        offline=wandb.offline,
    )
    # Add configuration to experiment
    wandb_logger.experiment.config.update(asdict(config))

    return wandb_logger
