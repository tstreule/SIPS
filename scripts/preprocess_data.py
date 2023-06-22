import json
from json import JSONDecodeError
from pathlib import Path
from warnings import warn

import typer

from sips.configs.base_config import _DatasetsConfig
from sips.configs.parsing import parse_train_file
from sips.data_extraction.preprocessing import (
    Preprocessor,
    call_preprocessing_steps,
    check_preprocessing_status,
    plot_overlap,
)

app = typer.Typer()


# TODO: put all the logic in a file in sips
@app.callback(invoke_without_command=True)
def main(
    config_path: str = "sips/configs/v0_dummy.yaml",
    extract: bool = True,
) -> None:
    config = parse_train_file(config_path).datasets
    rosbags = config.rosbags
    if extract:
        for this_rosbag in rosbags:
            print(100 * "=")
            print(f"Start processing {this_rosbag}")
            save_dir = call_preprocessing_steps(config, this_rosbag)
            if not save_dir is None:
                plot_overlap(config, source_dir=save_dir)
                continue

    return


if __name__ == "__main__":
    app()
