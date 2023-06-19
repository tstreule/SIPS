import json
from json import JSONDecodeError
from pathlib import Path
from warnings import warn

from sips.configs.base_config import _DatasetsConfig
from sips.configs.parsing import parse_train_file
from sips.data_extraction.preprocessing import Preprocessor
from sips.data_extraction.utils.handle_configs import find_config_num


def _check_preprocessing_status(
    config_num: int, source_dir: str | Path
) -> tuple[bool, bool, bool, bool, bool, bool]:
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
    # the process was run before and does not need to be repeated
    if "rosbag_connections_and_topics" in info:
        # if len(info["rosbag_connections_and_topics"]) > 0:
        conns_checked = True
    if (
        "num_extracted_sonar_datapoints" in info
        and "num_extracted_pose_datapoints" in info
    ):
        # if (
        #    info["num_extracted_sonar_datapoints"] > 0
        #    and info["num_extracted_pose_datapoints"] > 0
        # ):
        data_extracted = True
    if "num_matched_datapoints" in info:
        # if info["num_matched_datapoints"] > 0:
        data_matched = True
    if "num_datapoints_after_filtering" in info:
        # if info["num_datapoints_after_filtering"] > 0:
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


def call_preprocessing_steps(config: _DatasetsConfig, rosbag: str) -> None:
    source_dir = Path("data/filtered") / Path(rosbag).with_suffix("")
    config_num = find_config_num(config, source_dir)
    (
        conns_checked,
        data_extracted,
        data_matched,
        data_filtered,
        overlap_calculated,
        failed,
    ) = _check_preprocessing_status(config_num, source_dir)
    if failed:
        warn("This config failed in previous run, skipping the rest of the operations")
        return
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
    # Failed at some stage
    if not (
        conns_checked
        and data_extracted
        and data_matched
        and data_filtered
        and overlap_calculated
    ):
        config_num = find_config_num(config, source_dir)
        save_dir = source_dir / str(config_num)
        with open(save_dir / "info.json", "r") as f:
            try:
                info = json.load(f)
            except JSONDecodeError:
                info = {}
        with open(save_dir / "info.json", "w") as f:
            info["failed"] = True
            json.dump(info, f, indent=2)
    return


# TODO: put all the logic in a file in sips
def main(
    config_path: str = "sips/configs/v0_dummy.yaml",
    extract: bool = True,
    image_overlap: bool = True,
) -> None:
    config = parse_train_file(config_path).datasets
    rosbags = config.rosbags
    if extract:
        for this_rosbag in rosbags:
            print(100 * "=")
            print(f"Start processing {this_rosbag}")
            call_preprocessing_steps(config, this_rosbag)
            print()
    # if image_overlap:
    #     for this_rosbag in rosbags:
    #         image_overlap
    return


if __name__ == "__main__":
    main()
