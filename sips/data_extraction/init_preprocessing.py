import json
import random
from json import JSONDecodeError
from pathlib import Path
from warnings import warn

from sips.configs.base_config import _DatasetsConfig
from sips.data_extraction.preprocessing import Preprocessor
from sips.data_extraction.utils.handle_configs import find_config_num


def check_preprocessing_status(
    config_num: int, source_dir: str | Path
) -> tuple[bool, bool, bool, bool, bool, bool]:
    """
    Try to look into the info.json file for this config and rosbag and
    return a boolean value for each preprocessing step based on if
    the step was successfully ran before.

    Parameters
    ----------
    config_num: Name of the data subfolder based on the current configuration
    source_dir: Data directory of the current rosbag

    Returns
    ----------
    conns_checked : bool
        True if connections were checked before
    data_extracted : bool
        True if data was extracted before
    data_matched : bool
        True if data was matched before
    data_filtered : bool
        True if data was filtered before
    overlap_calculated : bool
        True if image overlap was calculated before
    failed : bool
        True if run failed before

    """
    if config_num == -1:
        return (False, False, False, False, False, False)
    source_dir = source_dir / Path(str(config_num))
    try:
        with open(source_dir / "info.json", "r") as f:
            info = json.load(f)
    except IOError:  # File is missing
        return (False, False, False, False, False, False)
    except JSONDecodeError:  # File is empty
        return (False, False, False, False, False, False)
    (
        conns_checked,
        data_extracted,
        data_matched,
        data_filtered,
        overlap_calculated,
        failed,
    ) = (
        False,
        False,
        False,
        False,
        False,
        False,
    )
    if "failed" in info:
        return (
            conns_checked,
            data_extracted,
            data_matched,
            data_filtered,
            overlap_calculated,
            True,
        )
    # Only check if the keys are available, if they are then
    # the process was run successfully before and does not need to be repeated
    if "rosbag_connections_and_topics" in info:
        conns_checked = True
    if (
        "num_extracted_sonar_datapoints" in info
        and "num_extracted_pose_datapoints" in info
    ):
        data_extracted = True
    if "num_matched_datapoints" in info:
        data_matched = True
    if "num_datapoints_after_filtering" in info:
        data_filtered = True
    if "num_tuples" in info:
        overlap_calculated = True
    return (
        conns_checked,
        data_extracted,
        data_matched,
        data_filtered,
        overlap_calculated,
        failed,
    )


def call_preprocessing_steps(config: _DatasetsConfig, rosbag: str) -> None | str | Path:
    """
    Run all preprocessing steps from here after making sure that they were not run before.
    Each step is only run if the steps before succeeded.

    Parameters
    ----------
    config: _DatasetsConfig
        current dataset configuration
    rosbag: str
        current rosbag

    Returns
    ----------
    save_dir: None | str | Path
        Directory were data for this configuration and rosbag is stored, only
        provided if all steps are successfull.

    """
    random.seed(config.seed)
    source_dir = Path("data/filtered") / Path(rosbag).with_suffix("")
    config_num = find_config_num(config, source_dir)
    (
        conns_checked,
        data_extracted,
        data_matched,
        data_filtered,
        overlap_calculated,
        failed,
    ) = check_preprocessing_status(config_num, source_dir)
    if failed:
        warn("This config failed in previous run, skipping the rest of the operations")
        return None
    if not (
        conns_checked
        and data_extracted
        and data_matched
        and data_filtered
        and overlap_calculated
    ):
        dataextraction = Preprocessor(config, rosbag)
        conns_checked = dataextraction.check_connections()
        if conns_checked:
            data_extracted = dataextraction.extract_data()
            if data_extracted:
                data_matched = dataextraction.match_data()
                if data_matched:
                    data_filtered = dataextraction.filter_data()
                    if data_filtered:
                        overlap_calculated = dataextraction.calculate_overlap()
    else:
        print(
            "Data has been preprocessed in previous run, skipping the rest of the operations"
        )
    config_num = find_config_num(config, source_dir)
    save_dir = source_dir / str(config_num)

    # Failed at some stage, store that info in info.json
    if not (
        conns_checked
        and data_extracted
        and data_matched
        and data_filtered
        and overlap_calculated
    ):
        with open(save_dir / "info.json", "r") as f:
            try:
                info = json.load(f)
            except JSONDecodeError:
                info = {}
        with open(save_dir / "info.json", "w") as f:
            info["failed"] = True
            json.dump(info, f, indent=2)
        return None
    return save_dir
