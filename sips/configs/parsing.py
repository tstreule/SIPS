from pathlib import Path

import yaml  # type: ignore[import]

from sips.configs.base_config import Config

__all__ = ["parse_train_file"]


def parse_train_file(file: str | Path) -> Config:
    """
    Parse file for training.

    Parameters
    ----------
    file : str | Path
        **.yaml** configuration file

    Returns
    -------
    Config
        Configuration object.

    """
    if Path(file).suffix == ".yaml":
        # Read .yaml file
        with open(file, "r") as f:
            config_dict = yaml.safe_load(f)
        assert isinstance(config_dict, dict)
        # Overwrite the default configuration with config_dict
        config = Config(**config_dict)
    else:
        raise ValueError("You need to provide a .yaml to train")

    return config
