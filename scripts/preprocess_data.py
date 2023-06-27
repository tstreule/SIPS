import json
from json import JSONDecodeError
from pathlib import Path
from warnings import warn

import typer

from sips.configs.base_config import _DatasetsConfig
from sips.configs.parsing import parse_train_file
from sips.data_extraction.init_preprocessing import (
    call_preprocessing_steps,
    check_preprocessing_status,
)
from sips.data_extraction.plotting import plot_overlap
from sips.data_extraction.preprocessing import Preprocessor

app = typer.Typer()


@app.callback(invoke_without_command=True)
def main(
    config_path: str = "sips/configs/v0_dummy.yaml",
    extract: bool = False,
    plot: bool = True,
) -> None:
    """
    Example script on how to run all preprocessing steps
    
    """
    config = parse_train_file(config_path).datasets
    rosbags = config.rosbags
    if extract:
        for this_rosbag in rosbags:
            print(100 * "=")
            print(f"Start processing {this_rosbag}")
            _ = call_preprocessing_steps(config, this_rosbag)
    if plot:
        for this_rosbag in rosbags:
            plot_overlap(config, this_rosbag)
            continue

    return


if __name__ == "__main__":
    app()
