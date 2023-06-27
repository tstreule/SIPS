import json
import os
from json import JSONDecodeError
from pathlib import Path

from sips.configs.base_config import _DatasetsConfig


# Add entry to summary.json file of this rosbag. If this configuration was looked
# at before just return its configuration number, else create new entry.
def add_entry_to_summary(config: _DatasetsConfig, rosbag: str | Path) -> int:
    save_dir = Path("data/filtered") / Path(rosbag).with_suffix("")
    variable_config_params = config.get_variable_params()
    # This is the first summary entry
    if not os.path.isfile(save_dir / "summary.json"):
        with open(save_dir / "summary.json", "w") as f:
            first_summary_entry = {0: variable_config_params}
            json.dump(first_summary_entry, f, indent=2)
        config_num = 0
        save_dir = save_dir / str(config_num)
        save_dir.mkdir(exist_ok=True)
        return int(config_num)
    with open(save_dir / "summary.json", "r") as f:
        summary = json.load(f)
    if variable_config_params in summary.values():
        for config_num, val in summary.items():
            if val == variable_config_params:
                save_dir = save_dir / str(config_num)
                save_dir.mkdir(exist_ok=True)
                return int(config_num)

    # Create new summary entry
    config_num = int(list(summary.keys())[-1]) + 1
    save_dir_config = save_dir / str(config_num)
    save_dir_config.mkdir(exist_ok=True)
    with open(save_dir / "summary.json", "w") as f:
        summary[config_num] = variable_config_params
        json.dump(summary, f, indent=2)
        return int(config_num)


# Find the configuration number according to this configuration
def find_config_num(config, source_dir) -> int:
    try:
        with open(source_dir / "summary.json", "r") as f:
            summary = json.load(f)
    except IOError:
        return -1
    except JSONDecodeError:
        return -1
    variable_config_params = config.get_variable_params()
    if variable_config_params in summary.values():
        for config_num, val in summary.items():
            if val == variable_config_params:
                return int(config_num)
    return -1
