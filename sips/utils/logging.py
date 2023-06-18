from pytorch_lightning.loggers.wandb import WandbLogger

from sips.configs.base_config import _WandBConfig


def setup_wandb_logger(config: _WandBConfig) -> bool | WandbLogger:
    if config.dry_run:
        return False
    return WandbLogger(
        name=config.name,
        save_dir=config.save_dir,
        version=config.version,
        project=config.project,
        entity=config.entity,
        offline=config.offline,
    )
