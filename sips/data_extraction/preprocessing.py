import itertools
import json
import os
from json import JSONDecodeError
from pathlib import Path
from warnings import warn

import cv2  # type: ignore
import numpy as np
import numpy.typing as npt
import torch
import torchvision.transforms as T
from PIL import Image, ImageFilter
from rosbags.rosbag1 import Reader
from rosbags.rosbag1.reader import ReaderError
from rosbags.serde import deserialize_cdr, ros1_to_cdr
from rosbags.typesys import get_types_from_msg, register_types
from rosbags.typesys.types import sensor_msgs__msg__Image as ImageMessage
from scipy.interpolate import interp1d
from torch import nn
from tqdm import tqdm

from sips.configs.base_config import _DatasetsConfig
from sips.data import CameraParams, CameraPose, SonarDatum
from sips.data_extraction.utils.handle_configs import (
    add_entry_to_summary,
    find_config_num,
)
from sips.utils.keypoint_matching import _unravel_index_2d
from sips.utils.point_projection import warp_image, warp_image_batch

# ==============================================================================
# Sonar image handling


def cosine_similarity(frame_normalized_1, frame_normalized_2) -> float:
    return np.sum(frame_normalized_1 * frame_normalized_2)


class RedundantImageFilter:
    """
    For filtering images that look too similar.
    """

    def __init__(
        self,
        blur: str | None = "gaussian",
        blur_size: float = 5,
        blur_std: float = 2.5,
        threshold: float = 0.95,
    ) -> None:
        # Some constants with values that will probably need some tuning
        assert blur in [None, "gaussian", "median", "bilateral"]
        assert 0 <= threshold <= 1, "Threshold must be within [0, 1]"
        self.blur = blur
        self.blur_size = blur_size
        self.blur_std = blur_std
        self.threshold = threshold  # Higher values will save more data, where 1.0 saves everything and 0.0 saves nothing.

        self.n_saved = 0
        self.n_redundant = 0
        self.current_frame_norm: npt.NDArray[np.float_] | None = None

    # TODO: adjust this such that it stores all images and keeps record of which images would
    # "survive" the filtering
    def image_redundant(
        self, message: ImageMessage, save_path: str | Path, save_ims: bool
    ) -> bool:
        keyframe = message.data.reshape((message.height, message.width))
        if save_ims:
            Image.fromarray(keyframe, mode="L").save(save_path)

        # Skip if image is redundant
        if self._is_keyframe_redundant(keyframe):
            self.n_redundant += 1
            return True

        # Save image
        self.n_saved += 1
        return False

    def _is_keyframe_redundant(self, frame: npt.NDArray[np.uint8]) -> bool:
        # Subsample the frame to a lower resolution for the sake of filtering
        frame_resized: npt.NDArray[np.uint8] = cv2.resize(frame, (512, 512))  # type: ignore

        # Blur the frame to prevent high frequency sonar noise to influence the frame
        # content distance measurements.  For reference on different blurring methods
        # see https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html
        frame_blurred: npt.NDArray[np.uint8]
        if self.blur is None:
            frame_blurred = frame_resized
        elif self.blur == "gaussian":
            frame_blurred = cv2.GaussianBlur(
                frame_resized, (self.blur_size, self.blur_size), self.blur_std
            )
        elif self.blur == "median":
            frame_blurred = cv2.medianBlur(frame_resized, self.blur_size)
        elif self.blur == "bilateral":
            frame_blurred = cv2.bilateralFilter(frame_resized, 9, 75, 75)
        else:
            raise ValueError(f"Invalid blur type '{self.blur}'")

        # L2 normalization of the frame as a vector, to prepare for cosine similarity
        frame_norm = frame_blurred / np.linalg.norm(frame_blurred)
        if (
            self.current_frame_norm is None
            or cosine_similarity(self.current_frame_norm, frame_norm) < self.threshold
        ):
            # Frame is not redundant
            self.current_frame_norm = frame_norm
            return False
        else:
            # Frame is redundant
            return True


def check_preprocessing_status(
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


def call_preprocessing_steps(config: _DatasetsConfig, rosbag: str) -> None | str | Path:
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
    config_num = find_config_num(config, source_dir)
    save_dir = source_dir / str(config_num)
    # Failed at some stage
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
    return save_dir


class Preprocessor:
    """
    Extract, match and filter the data
    """

    def __init__(self, config: _DatasetsConfig, rosbag: str | Path) -> None:
        self.rosbag = rosbag
        self.rosbag_path = Path("data/raw") / rosbag
        self.save_dir = Path("data/filtered") / Path(rosbag).with_suffix("")
        self.image_save_dir = self.save_dir / "images"
        self.image_filter_params: dict[str, float | str | None] = {
            "blur": config.image_filter,
            "blur_size": config.image_filter_size,
            "blur_std": config.image_filter_std,
            "threshold": config.image_filter_threshold,
        }
        self.sonar_params: dict[str, float] = {
            "horizontal_fov": config.horizontal_fov,
            "vertical_fov": config.vertical_fov,
            "max_range": config.max_range,
            "min_range": config.min_range,
        }
        self.conv_size = config.conv_size
        self.bs_threshold = config.bright_spots_threshold
        self.n_elevations = config.n_elevations
        self.overlap_threshold = config.overlap_ratio_threshold
        self.im_shape = config.image_shape

        self.poses: list[dict[str, int | dict[str, float] | list[float]]] = []
        self.unfiltered_poses: list[
            dict[str, int | dict[str, float] | list[float]]
        ] = []
        self.sonar_images: dict[int, ImageMessage] = {}
        self.sonar_datum: list[SonarDatum] = []
        self.tuple_stamps: list[tuple[int, int]] = []

        self.rosbag_connections: list[str] = []
        self.num_extracted_sonar = 0
        self.num_extracted_pose = 0
        self.num_matched_datapoints = 0
        self.num_datapoints_after_filtering = 0

        with open("sips/data_extraction/misc/SonarConfiguration.msg", "r") as f:
            SONAR_CONFIGURATION_MSG = f.read()

        register_types(
            get_types_from_msg(
                SONAR_CONFIGURATION_MSG,
                "sonar_driver_interfaces/msg/SonarConfiguration",
            )
        )
        from rosbags.typesys.types import (  # type: ignore  # noqa
            sonar_driver_interfaces__msg__SonarConfiguration as SonarConfiguration,
        )

        self.save_dir.mkdir(exist_ok=True, parents=True)
        config_num = add_entry_to_summary(config, rosbag)
        if config_num == -1:
            warn(
                "No configuration number was assigned, skipping the rest of the operations"
            )
        self.this_config_save_dir = self.save_dir / str(config_num)
        self.this_config_save_dir.mkdir(exist_ok=True, parents=True)
        if not os.path.isfile(self.this_config_save_dir / "info.json"):
            with open(self.this_config_save_dir / "info.json", "w") as f:
                pass

    def _update_info(
        self,
        key: str,
        value: int | list[str] | dict[str, float | str | None] | dict[str, float],
    ) -> None:
        with open(self.this_config_save_dir / "info.json", "r") as f:
            try:
                info = json.load(f)
            except JSONDecodeError:
                info = {}
        with open(self.this_config_save_dir / "info.json", "w") as f:
            info[key] = value
            json.dump(info, f, indent=2)
        return

    # ==============================================================================
    # Match interpolated pose data with sonar timestamps

    def _match_pose_data(self, sonar_stamps, poses):
        pose_interp = self._make_pose_interpolator(poses)
        pose_data_matched = []
        for sonar_stamp in tqdm(sonar_stamps):
            interpolated_pose = pose_interp(sonar_stamp)
            pose: dict[str, int | dict[str, float]] = {
                "timestamp": sonar_stamp,  # type: ignore
                "point_position": {
                    "x": interpolated_pose[0],
                    "y": interpolated_pose[1],
                    "z": interpolated_pose[2],
                },
                "quaterion_orientation": {
                    "x": interpolated_pose[3],
                    "y": interpolated_pose[4],
                    "z": interpolated_pose[5],
                    "w": interpolated_pose[6],
                },
                "covariance": interpolated_pose[7:].tolist(),
            }
            pose_data_matched.append(pose)
        return pose_data_matched

    # ==============================================================================
    # Interpolators for both sonar and twist data

    def _make_pose_interpolator(
        self, poses: list[dict[str, int | dict[str, float]]]
    ) -> interp1d:
        timestamps = [pose["timestamp"] for pose in poses]

        # Alternatively you can write the following when you want the timestamps in seconds
        # timestamps = [pose["timestamp"] * 1e-9 for pose in poses]
        locations = np.array(
            [[pose["point_position"][key] for key in "xyz"] for pose in poses]  # type: ignore
        ).T
        quaternions = np.array(
            [[pose["quaterion_orientation"][key] for key in "xyzw"] for pose in poses]  # type: ignore
        ).T
        covariances = np.array([pose["covariance"] for pose in poses]).T
        return interp1d(
            timestamps, np.concatenate([locations, quaternions, covariances], axis=0)
        )

    # TODO: check if pose recording frequency and other recording parameters are correct
    def check_connections(self) -> bool:
        sonar_topic = "/proteus_PRA01/sonar_0/image"
        pose_topic = "/proteus_PRA01/gtsam/pose"
        sonar_topic_available = False
        pose_topic_available = False
        rosbag_connections: list[str] = []
        try:
            with Reader(self.rosbag_path) as reader:
                for connection in reader.connections:
                    if connection.topic == sonar_topic:
                        sonar_topic_available = True
                    elif connection.topic == pose_topic:
                        pose_topic_available = True
                    rosbag_connections.append(
                        f"{connection.topic} {connection.msgtype}"
                    )
        except ReaderError:
            warn("Rosbag file missing, skipping the rest of the operations")
            return False
        if not (sonar_topic_available and pose_topic_available):
            warn(
                "Sonar topic and/or pose topic is missing within this bag, skipping the rest of the operations"
            )
            return False

        self._update_info("rosbag_connections_and_topics", rosbag_connections)
        self.rosbag_connections = rosbag_connections
        return True

    # TODO: check if the sonar configuration parameters match the once given in the config
    # if not given, ignore and if not matching throw a warning as they should be consistent
    # for the model to get sonar images of the same type
    def extract_data(self) -> bool:
        print(f"Extract data")
        # Read data from rosbag file
        try:
            with Reader(self.rosbag_path) as reader:
                # --------

                sonar_topic = "/proteus_PRA01/sonar_0/image"
                pose_topic = "/proteus_PRA01/gtsam/pose"
                sonar_config_topic = "/proteus_PRA01/sonar_0/configuration"
                sonar_connections = [
                    x for x in reader.connections if x.topic == sonar_topic
                ]
                pose_connections = [
                    x for x in reader.connections if x.topic == pose_topic
                ]
                if len(sonar_connections) < 1 or len(pose_connections) < 1:
                    warn(
                        "No pose data or sonar data found in this bag, skipping the rest of the operations"
                    )
                    self._update_info("num_extracted_sonar_datapoints", 0)
                    self._update_info("num_extracted_pose_datapoints", 0)
                    return False
                sonar_config_connections = [
                    x for x in reader.connections if x.topic == sonar_config_topic
                ]
                if len(sonar_config_connections) < 1:
                    warn(
                        "No Sonar configuration data found in this bag, skipping the rest of the operations"
                    )
                    self._update_info("num_extracted_sonar_datapoints", 0)
                    self._update_info("num_extracted_pose_datapoints", 0)
                    return False

                connections = [
                    x
                    for x in reader.connections
                    if x.topic in [sonar_topic, pose_topic, sonar_config_topic]
                ]
                config_msgs = []  # TODO: remove this again, only used for some tests
                first_pose_seen = False
                for connection, timestamp, rawdata in reader.messages(connections):
                    # Read image message and save
                    msg = deserialize_cdr(
                        ros1_to_cdr(rawdata, connection.msgtype), connection.msgtype
                    )

                    if (
                        connection.msgtype == "sensor_msgs/msg/Image"
                        and first_pose_seen
                    ):
                        timestamp = int(
                            f"{msg.header.stamp.sec}{msg.header.stamp.nanosec:09}"
                        )
                        self.sonar_images[timestamp] = msg

                    if (
                        connection.msgtype
                        == "geometry_msgs/msg/PoseWithCovarianceStamped"
                    ):
                        first_pose_seen = True
                        pp = msg.pose.pose.position
                        po = msg.pose.pose.orientation
                        pose: dict[str, int | dict[str, float] | list[float]] = {
                            "timestamp": int(f"{msg.header.stamp.sec}{msg.header.stamp.nanosec:09}"),  # type: ignore
                            "point_position": {"x": pp.x, "y": pp.y, "z": pp.z},
                            "quaterion_orientation": {
                                "x": po.x,
                                "y": po.y,
                                "z": po.z,
                                "w": po.w,
                            },
                            "covariance": msg.pose.covariance.tolist(),
                        }
                        self.poses.append(pose)

                    # TODO: sonar_params should be initialized through the config file
                    # if the params do not coincide with them throw a warning
                    # should probably continue with the operation and save data whenever
                    # the sonar params match the ones from the config again, but seems
                    # cumbersome

                    if (
                        connection.msgtype
                        == "sonar_driver_interfaces/msg/SonarConfiguration"
                    ):
                        config_msgs.append(msg)
                        if (
                            self.sonar_params["horizontal_fov"] != msg.horz_fov
                            or self.sonar_params["vertical_fov"] != msg.vert_fov
                            or self.sonar_params["max_range"] != msg.max_range
                            or self.sonar_params["min_range"] != msg.min_range
                        ):
                            warn(
                                "Sonar parameters do not match the ones given in the config file, skipping the rest of the operations"
                            )
                            self._update_info("num_extracted_sonar_datapoints", 0)
                            self._update_info("num_extracted_pose_datapoints", 0)
                            return False

                # Check the frequency of the pose data recordings, if they are too high
                # we ignore this bag as the accuracy is lower given measurement drifts
                pose_frequency = 1 / (
                    (
                        (self.poses[-1]["timestamp"] - self.poses[0]["timestamp"])  # type: ignore
                        / len(self.poses)
                    )
                    / 1_000_000_000
                )
                if pose_frequency > 5:
                    warn(
                        f"Frequency of pose data too high ({pose_frequency:.2f}Hz)"
                        ", skipping the rest of the operations"
                    )
                    self._update_info("num_extracted_sonar_datapoints", 0)
                    self._update_info("num_extracted_pose_datapoints", 0)
                    return False

                num_extracted_sonar = len(self.sonar_images)
                num_extracted_pose = len(self.poses)
                if num_extracted_pose <= 0 or num_extracted_sonar <= 0:
                    warn(
                        "No pose and/or sonar data extracted, skipping the rest of the operations"
                    )
                    self._update_info(
                        "num_extracted_sonar_datapoints", num_extracted_sonar
                    )
                    self._update_info(
                        "num_extracted_pose_datapoints", num_extracted_pose
                    )
                    return False
                print(f"Number of extracted sonar images:   {num_extracted_sonar:>7}")
                print(f"Number of extracted pose data:   {num_extracted_pose:>7}")
                print()
                self._update_info("num_extracted_sonar_datapoints", num_extracted_sonar)
                self._update_info("num_extracted_pose_datapoints", num_extracted_pose)
                self.num_extracted_sonar = num_extracted_sonar
                self.num_extracted_pose = num_extracted_pose
                return True
        except ReaderError:
            warn("Rosbag file missing, skipping the rest of the operations")
        return False

    def match_data(
        self,
    ) -> (
        bool
    ):  # TODO: kind of a nameclash with function above that also includes "match" -> change one
        print("Match sonar data with poses through interpolation")

        # Read the sonar timestamps in ascending order
        sonar_stamps = sorted(list(self.sonar_images.keys()))
        sonar_stamps_matched: list[int] = []
        sonar_stamps_unmatched: list[int] = []

        # First sonar datapoint should always happen after first pose datapoint
        # to improve accuracy of interpolation. The last sonar datapoint should
        # always happen before the last pose datapoint for the same reason
        assert (
            self.poses[0]["timestamp"] < sonar_stamps[0]  # type: ignore[operator]
        ), "First sonar datapoint before first pose datapoint"
        if self.poses[-1]["timestamp"] < sonar_stamps[-1]:  # type: ignore[operator]
            # Last sonar datapoint after last pose datapoint, removing sonar data after last pose datapoint
            for i in range(2, len(sonar_stamps)):
                if sonar_stamps[-i] <= self.poses[-1]["timestamp"]:  # type: ignore[operator]
                    sonar_stamps_matched = sonar_stamps[: -i + 1]
                    sonar_stamps_unmatched = sonar_stamps[-i + 1 :]
                    break
        else:
            sonar_stamps_matched = sonar_stamps
            sonar_stamps_unmatched = []
        if len(sonar_stamps_matched) <= 0:
            warn(
                "No remaining sonar data found after removing sonar images recorded after last pose datapoint, skipping the rest of the operations"
            )
            self._update_info("num_matched_datapoints", len(sonar_stamps_matched))
            return False
        assert (
            self.poses[-1]["timestamp"] >= sonar_stamps_matched[-1]  # type: ignore[operator]
        ), "Last sonar datapoint after last pose datapoint"

        # Remove the sonar images that were recorded after the last pose
        # datapoint
        for unmatch in sonar_stamps_unmatched:
            self.sonar_images.pop(unmatch)

        # Create interpolation over all pose and twist datapoints
        # then find interpolated pose and twist datapoint for each
        # sonar timestamp
        self.poses = self._match_pose_data(sonar_stamps_matched, self.poses)
        assert len(self.poses) == len(
            self.sonar_images
        ), "length of matched data does not match"
        num_matched_datapoints = len(self.sonar_images)
        print(f"Number of matched datapoints:   {num_matched_datapoints:>7}")
        print()
        self._update_info("num_matched_datapoints", num_matched_datapoints)
        self.num_matched_datapoints = num_matched_datapoints
        return True

    def filter_data(self) -> bool:
        print("Filter sonar images with redudant image filter")

        (self.image_save_dir).mkdir(exist_ok=True)

        image_filter = RedundantImageFilter(**self.image_filter_params)  # type: ignore[arg-type]

        # if no images have been saved for previous configs, save in this run
        save_ims = len(os.listdir(self.image_save_dir)) != self.num_matched_datapoints

        sonar_timestamps = sorted(list(self.sonar_images.keys()))
        for pose, sonar_timestamp in tqdm(zip(self.poses, sonar_timestamps)):
            assert (
                pose["timestamp"] == sonar_timestamp
            ), "pose timestamp and sonar timestamp not agreeing"
            fname = f"sonar_{sonar_timestamp}.png"
            img = self.sonar_images[sonar_timestamp]
            image_redundant = image_filter.image_redundant(
                img, self.image_save_dir / fname, save_ims
            )
            if not image_redundant:
                self.unfiltered_poses.append(pose)
        assert len(self.unfiltered_poses) == image_filter.n_saved
        num_datapoints_after_filtering = len(self.unfiltered_poses)
        if num_datapoints_after_filtering <= 0:
            warn(
                "No remaining sonar data found after applying redundant image filter, skipping the rest of the operations"
            )
            self._update_info(
                "num_datapoints_after_filtering", num_datapoints_after_filtering
            )
            return False
        with open(self.this_config_save_dir / "unfiltered_pose_data.json", "w") as f:
            json.dump(self.unfiltered_poses, f, indent=2)
        with open(self.save_dir / "pose_data.json", "w") as f:
            json.dump(self.poses, f, indent=2)
        print(
            f"Number of datapoints after filtering:   {num_datapoints_after_filtering:>7}"
        )
        print()
        self._update_info(
            "num_datapoints_after_filtering", num_datapoints_after_filtering
        )
        self.num_datapoints_after_filtering = num_datapoints_after_filtering
        print(
            f"Data extraction steps are finished, data can be found in {self.save_dir} and {self.this_config_save_dir}"
        )
        print()
        return True

    def _blurred_tensor(self, im: Image.Image) -> torch.Tensor:
        transform = T.Compose([T.PILToTensor()])
        return transform((im.resize(self.im_shape)).filter(ImageFilter.BLUR))

    # Find pixels that are brighter than threshold. Only keep one per 8x8
    # grid. Return as uv coordinates
    def _find_bright_spots(self, im: torch.Tensor) -> torch.Tensor:
        maxpool = nn.MaxPool2d(
            self.conv_size, stride=self.conv_size, return_indices=True
        )
        pool, indices = maxpool(im.to(torch.float))
        mask = pool > self.bs_threshold
        masked_indices = indices[mask]
        row, col = _unravel_index_2d(masked_indices, [4096, 512])
        bright_spots_temp = torch.column_stack((row, col)).to(torch.float)
        bright_spots = torch.full((64 * 64, 2), torch.nan)
        bright_spots[: bright_spots_temp.shape[0], :] = bright_spots_temp
        return bright_spots.permute(1, 0).view(2, 64, 64)

    def _overlap_ratio_sufficient(
        self,
        bs1_uv: torch.Tensor,
        bs2_uv: torch.Tensor,
        sonar_datum1: SonarDatum,
        sonar_datum2: SonarDatum,
        num_bs1: int,
        num_bs2: int,
    ) -> bool:
        height, width = sonar_datum1.image.shape[1:]
        assert (
            sonar_datum1.image.shape == sonar_datum2.image.shape
        ), "Image shapes do not match"
        bs1_uv_proj = warp_image_batch(
            bs1_uv.unsqueeze(0),
            [sonar_datum1.params],
            [sonar_datum1.pose],
            [sonar_datum2.params],
            [sonar_datum2.pose],
            (width, height),
        )
        bs2_uv_proj = warp_image_batch(
            bs2_uv.unsqueeze(0),
            [sonar_datum2.params],
            [sonar_datum2.pose],
            [sonar_datum1.params],
            [sonar_datum1.pose],
            (width, height),
        )
        num_bs1_proj = (~bs1_uv_proj.isnan()).sum() / 2
        num_bs2_proj = (~bs2_uv_proj.isnan()).sum() / 2
        ratio1 = num_bs1_proj / num_bs1
        ratio2 = num_bs2_proj / num_bs2
        return float((ratio1.float() + ratio2.float()) / 2) > self.overlap_threshold

    def calculate_overlap(self) -> bool:
        print("Calculate the image overlap of image pairs")
        camera_params = CameraParams(
            min_range=self.sonar_params["min_range"],
            max_range=self.sonar_params["max_range"],
            azimuth=self.sonar_params["horizontal_fov"],
            elevation=self.sonar_params["vertical_fov"],
        )
        for this_pose in self.unfiltered_poses:
            stamp = int(this_pose["timestamp"])  # type:ignore[arg-type]
            im_path = self.image_save_dir / f"sonar_{stamp}.png"
            try:
                im = Image.open(im_path)
            except FileNotFoundError:
                warn("Image file not found, skipping the rest of the operations")
                return False
            im_tensor = self._blurred_tensor(im)
            camera_pose = CameraPose(
                position=[
                    this_pose["point_position"]["x"],  # type: ignore[call-overload, index]
                    this_pose["point_position"]["y"],  # type: ignore[call-overload, index]
                    this_pose["point_position"]["z"],  # type: ignore[call-overload, index]
                ],
                rotation=[
                    this_pose["quaterion_orientation"]["x"],  # type: ignore[call-overload, index]
                    this_pose["quaterion_orientation"]["y"],  # type: ignore[call-overload, index]
                    this_pose["quaterion_orientation"]["z"],  # type: ignore[call-overload, index]
                    this_pose["quaterion_orientation"]["w"],  # type: ignore[call-overload, index]
                ],
            )
            self.sonar_datum.append(
                SonarDatum(
                    image=im_tensor, pose=camera_pose, params=camera_params, stamp=stamp
                )
            )
        if len(self.sonar_datum) < 2:
            warn("Not enough sonar data found, skipping the rest of the operations")
            return False
        bright_spots: dict[int, torch.Tensor] = {}
        num_bright_spots: dict[int, int] = {}
        for sonar_datum in self.sonar_datum:
            bs_uv = self._find_bright_spots(sonar_datum.image)
            bright_spots[sonar_datum.stamp] = bs_uv
            num_bright_spots[sonar_datum.stamp] = int(
                (~bs_uv.isnan()).sum() * self.n_elevations / 2
            )
        tuple_stamps: list[tuple[int, int]] = []
        for sonar_datum1, sonar_datum2 in tqdm(
            itertools.combinations(self.sonar_datum, 2)
        ):
            if self._overlap_ratio_sufficient(
                bright_spots[sonar_datum1.stamp],
                bright_spots[sonar_datum2.stamp],
                sonar_datum1,
                sonar_datum2,
                num_bright_spots[sonar_datum1.stamp],
                num_bright_spots[sonar_datum2.stamp],
            ):
                tuple_stamps.append(
                    (sonar_datum1.stamp, sonar_datum2.stamp)
                )  # Q: is it ok that this i symmetric?
                tuple_stamps.append((sonar_datum2.stamp, sonar_datum1.stamp))

        num_tuples = len(tuple_stamps)
        num_bs_mean = np.array([num_bs for num_bs in num_bright_spots.values()]).mean()
        self._update_info("num_tuples", num_tuples)
        print(f"Number of tuples with sufficient overlap:   {num_tuples:>7}")
        print(f"Average number of bright spots detected per image:  {num_bs_mean:>7}")
        with open(self.this_config_save_dir / "tuple_stamps.json", "w") as f:
            json.dump(tuple_stamps, f, indent=2)
        return True
