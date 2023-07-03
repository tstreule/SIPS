from dataclasses import asdict
from typing import Literal

from pytorch_lightning.loggers.wandb import WandbLogger

import wandb
from sips.configs.base_config import Config
from sips.utils.dotenv import _get_file_parent_dir


def setup_wandb_logger(config: Config) -> Literal[False] | WandbLogger:
    # Dry run / no logging
    if config.wandb.dry_run:
        return False

    # Init logger
    code_dir = str(_get_file_parent_dir("sips").parent)
    wandb_logger = WandbLogger(
        name=config.wandb.name,
        save_dir=config.wandb.save_dir,
        version=config.wandb.version,
        project=config.wandb.project,
        entity=config.wandb.entity,
        offline=config.wandb.offline,
        log_model="all",  # synched with ModelCheckpoint callbacks
        settings=wandb.Settings(code_dir=code_dir),
    )
    # Add configuration to experiment
    wandb_logger.experiment.config.update(asdict(config))

    return wandb_logger
