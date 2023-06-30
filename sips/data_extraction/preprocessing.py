import itertools
import json
import os
import random
from json import JSONDecodeError
from pathlib import Path
from warnings import warn

import numpy as np
import torch
from PIL import Image
from rosbags.rosbag1 import Reader
from rosbags.rosbag1.reader import ReaderError
from rosbags.serde import deserialize_cdr, ros1_to_cdr
from rosbags.typesys import get_types_from_msg, register_types
from rosbags.typesys.types import sensor_msgs__msg__Image as ImageMessage
from scipy.interpolate import interp1d
from tqdm import tqdm

from sips.configs.base_config import _DatasetsConfig
from sips.data import CameraParams, CameraPose, SonarDatum
from sips.data_extraction.image_filter import RedundantImageFilter
from sips.data_extraction.utils.handle_configs import add_entry_to_summary
from sips.data_extraction.utils.image_operations import (
    blur_image,
    find_bright_spots,
    image_to_tensor,
)
from sips.utils.point_projection import warp_image_batch


class Preprocessor:
    """
    Extract, match and filter the data and create (pose, sonar image) tuples
    """

    def __init__(self, config: _DatasetsConfig, rosbag: str | Path) -> None:
        self.rosbag = rosbag  # rosbag name of form <rosbag-name>.bag
        self.rosbag_path = Path("data/raw") / rosbag  # path to the raw rosbag data
        self.save_dir = Path("data/filtered") / Path(rosbag).with_suffix(
            ""
        )  # path where to store data
        self.image_save_dir = self.save_dir / "images"  # path where to store image data
        self.image_filter_params: dict[
            str, float | str | None
        ] = {  # redundant image filter params
            "blur": config.image_filter,
            "blur_size": config.image_filter_size,
            "blur_std": config.image_filter_std,
            "threshold": config.image_filter_threshold,
        }
        self.sonar_params: dict[str, float] = {  # sonar configuration params
            "horizontal_fov": config.horizontal_fov,
            "vertical_fov": config.vertical_fov,
            "max_range": config.max_range,
            "min_range": config.min_range,
        }
        self.conv_size = config.conv_size  # Convolution size for bright spot detection
        self.bs_threshold = (
            config.bright_spots_threshold
        )  # Bright spot detection threshold
        self.n_elevations = (
            config.n_elevations
        )  # Number of elevation for pixel projection between images
        self.overlap_threshold = (
            config.overlap_ratio_threshold
        )  # Threshold for ratio of image overlap
        self.im_shape = config.image_shape  # image shape

        self.all_poses: list[
            dict[str, int | dict[str, float] | list[float]]
        ] = []  # Pose data
        self.unfiltered_poses: list[
            dict[str, int | dict[str, float] | list[float]]
        ] = []  # Remaining pose data after filtering
        self.sonar_images: dict[
            int, ImageMessage
        ] = {}  # Sonar images, keyed by timestamp
        self.tuple_stamps: list[
            tuple[int, int]
        ] = []  # timestamps of tuples with sufficient overlap

        self.num_matched_datapoints = (
            0  # Number of datapoints after matching pose and sonar data
        )

        # Define Sonar configuration message
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
        # Find configuration number for this rosbag and config combination and create
        # folder structure for it.
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
        """
        Write number of results of each step to corresponding info.json file

        """
        with open(self.this_config_save_dir / "info.json", "r") as f:
            try:
                info = json.load(f)
            except JSONDecodeError:
                info = {}
        with open(self.this_config_save_dir / "info.json", "w") as f:
            info[key] = value
            json.dump(info, f, indent=2)
        return

    def check_connections(self) -> bool:
        """
        Go through all available connections of current bag

        Connections for pose data, sonar images and sonar configuration have to be available

        Returns
        -------
        bool
            indicating success of operations

        """
        sonar_topic = "/proteus_PRA01/sonar_0/image"
        pose_topic = "/proteus_PRA01/gtsam/pose"
        sonar_config_topic = "/proteus_PRA01/sonar_0/configuration"
        sonar_topic_available = False
        pose_topic_available = False
        sonar_config_topic_available = False
        rosbag_connections: list[str] = []
        try:
            with Reader(self.rosbag_path) as reader:
                for connection in reader.connections:
                    if connection.topic == sonar_topic:
                        sonar_topic_available = True
                    elif connection.topic == pose_topic:
                        pose_topic_available = True
                    elif connection.topic == sonar_config_topic:
                        sonar_config_topic_available = True
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
        if not sonar_config_topic_available:
            warn(
                "Sonar configuration topic is missing within this bag, skipping the rest of the operations"
            )
            return False

        self._update_info("rosbag_connections_and_topics", rosbag_connections)
        return True

    # ==============================================================================
    # Data extraction

    def extract_data(self) -> bool:
        """
        Extract data from current rosbag file

        Checks that pose data, sonar imagery data and sonar configuration data is available and applicable.
        For sonar configuration we only keep data if it matches sonar configuration of given config file.
        Pose data has to have a low frequency (4Hz) as higher frequencies lead to lower accuracy coming
        from measuring drift. Sonar images are only saved after we saw first pose datapoint to improve
        the accuracy of later interpolation steps.


        Returns
        -------
        bool
            indicating success of operations

        """
        print(f"Extract data")

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
                first_pose_seen = False
                for connection, timestamp, rawdata in reader.messages(connections):
                    msg = deserialize_cdr(
                        ros1_to_cdr(rawdata, connection.msgtype), connection.msgtype
                    )

                    if (
                        connection.msgtype == "sensor_msgs/msg/Image"
                        and first_pose_seen
                    ):
                        # msg contains sonar image
                        timestamp = int(
                            f"{msg.header.stamp.sec}{msg.header.stamp.nanosec:09}"
                        )
                        self.sonar_images[timestamp] = msg

                    if (
                        connection.msgtype
                        == "geometry_msgs/msg/PoseWithCovarianceStamped"
                    ):
                        # msg contains pose data
                        first_pose_seen = True
                        pp = msg.pose.pose.position
                        po = msg.pose.pose.orientation
                        pose: dict[str, int | dict[str, float] | list[float]] = {
                            "timestamp": int(
                                f"{msg.header.stamp.sec}{msg.header.stamp.nanosec:09}"
                            ),
                            "point_position": {"x": pp.x, "y": pp.y, "z": pp.z},
                            "quaterion_orientation": {
                                "x": po.x,
                                "y": po.y,
                                "z": po.z,
                                "w": po.w,
                            },
                            "covariance": msg.pose.covariance.tolist(),
                        }
                        self.all_poses.append(pose)

                    if (
                        connection.msgtype
                        == "sonar_driver_interfaces/msg/SonarConfiguration"
                    ):
                        # msg contains sonar configuration data
                        if (  # sonar configuration has to match values given in config file
                            self.sonar_params["horizontal_fov"] != msg.horz_fov
                            or self.sonar_params["vertical_fov"] != msg.vert_fov
                            or self.sonar_params["max_range"] != msg.max_range
                            or self.sonar_params["min_range"] != msg.min_range
                        ):
                            warn(
                                "Sonar configuration parameters do not match the ones given in the config file, skipping the rest of the operations"
                            )
                            self._update_info("num_extracted_sonar_datapoints", 0)
                            self._update_info("num_extracted_pose_datapoints", 0)
                            return False

                # Check the frequency of the pose data recordings, if they are too high
                # we ignore this bag as the accuracy is lower given measurement drifts
                pose_frequency = 1 / (
                    (
                        (self.all_poses[-1]["timestamp"] - self.all_poses[0]["timestamp"])  # type: ignore[operator]
                        / len(self.all_poses)
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
                num_extracted_pose = len(self.all_poses)
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
                return True
        except ReaderError:
            warn("Rosbag file missing, skipping the rest of the operations")
        return False

    # ==============================================================================
    # Pose and sonar datapoint matching

    def _make_pose_interpolator(
        self, poses: list[dict[str, int | dict[str, float] | list[float]]]
    ) -> interp1d:
        """
        Create the interpolator based on all pose data

        """
        timestamps = [pose["timestamp"] for pose in poses]

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

    def _interpolate_pose_data(
        self,
        sonar_stamps: list[int],
        poses: list[dict[str, int | dict[str, float] | list[float]]],
    ):
        """
        Given the timestamps of the sonar datapoints, find interpolated poses for the timestamps

        Parameters
        ----------
        sonar_stamps: all timestamps of the remaining sonar data from this bag
        poses: all pose datapoints, needed to create interpolator

        Returns
        ----------
        pose_data_matched : list[dict[str, int | dict[str, float] | list[float]]]
            interpolated poses according to the sonar timestamps

        """
        pose_interp = self._make_pose_interpolator(poses)
        pose_data_matched = []
        for this_sonar_stamp in tqdm(sonar_stamps):
            interpolated_pose = pose_interp(this_sonar_stamp)
            pose: dict[str, int | dict[str, float]] = {
                "timestamp": this_sonar_stamp,
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

    def match_data(
        self,
    ) -> bool:
        """
        For each sonar datapoint we interpolate a pose datapoint at the same timestamp.
        Remove sonar data after last pose datapoint to improve the accuracy of the
        interpolation.

        Returns
        -------
        bool
            indicating success of operations

        """
        print("Match sonar data with poses through interpolation")

        # Read the sonar timestamps in ascending order
        sonar_stamps = sorted(list(self.sonar_images.keys()))
        sonar_stamps_matched: list[int] = []
        sonar_stamps_unmatched: list[int] = []

        # First sonar datapoint should always happen after first pose datapoint
        # to improve accuracy of interpolation. The last sonar datapoint should
        # always happen before the last pose datapoint for the same reason
        assert (
            self.all_poses[0]["timestamp"] <= sonar_stamps[0]  # type: ignore[operator]
        ), "First sonar datapoint before first pose datapoint"
        if self.all_poses[-1]["timestamp"] < sonar_stamps[-1]:  # type: ignore[operator]
            # Last sonar datapoint after last pose datapoint, removing sonar data after last pose datapoint
            for i in range(2, len(sonar_stamps)):
                if sonar_stamps[-i] <= self.all_poses[-1]["timestamp"]:  # type: ignore[operator]
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
            self.all_poses[-1]["timestamp"] >= sonar_stamps_matched[-1]  # type: ignore[operator]
        ), "Last sonar datapoint after last pose datapoint"

        # Remove the sonar images that were recorded after the last pose
        # datapoint
        for unmatch in sonar_stamps_unmatched:
            self.sonar_images.pop(unmatch)

        # Create interpolation over all pose and twist datapoints
        # then find interpolated pose and twist datapoint for each
        # sonar timestamp
        self.all_poses = self._interpolate_pose_data(
            sonar_stamps_matched, self.all_poses
        )
        assert len(self.all_poses) == len(
            self.sonar_images
        ), "length of matched data does not match"
        num_matched_datapoints = len(self.sonar_images)
        print(f"Number of matched datapoints:   {num_matched_datapoints:>7}")
        print()
        self._update_info("num_matched_datapoints", num_matched_datapoints)
        self.num_matched_datapoints = num_matched_datapoints
        return True

    # ==============================================================================
    # Filter data

    def filter_data(self) -> bool:
        """
        Filter the sonar data with the redundant image filter. Store all remaining images
        even the ones filtered out if not done in previous run. Information of what sonar
        data remains after filtering can be found in pose timestamps.

        Returns
        -------
        bool
            indicating success of operations

        """
        print("Filter sonar images with redudant image filter")

        (self.image_save_dir).mkdir(exist_ok=True)

        image_filter = RedundantImageFilter(**self.image_filter_params)  # type: ignore[arg-type]

        # if no images have been saved for previous configs, save in this run
        save_ims = len(os.listdir(self.image_save_dir)) != self.num_matched_datapoints

        # Store all sonar images and filter out redundant datapoints
        sonar_timestamps = sorted(list(self.sonar_images.keys()))
        for pose, sonar_timestamp in tqdm(zip(self.all_poses, sonar_timestamps)):
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
        assert (
            len(self.unfiltered_poses) == image_filter.n_saved
        ), "number of unfiltered poses not agreeing with number of saved images"
        num_datapoints_after_filtering = len(self.unfiltered_poses)
        if num_datapoints_after_filtering <= 0:
            warn(
                "No remaining sonar data found after applying redundant image filter, skipping the rest of the operations"
            )
            self._update_info(
                "num_datapoints_after_filtering", num_datapoints_after_filtering
            )
            return False

        # Store the pose data
        with open(self.this_config_save_dir / "unfiltered_pose_data.json", "w") as f:
            json.dump(self.unfiltered_poses, f, indent=2)

        print(
            f"Number of datapoints after filtering:   {num_datapoints_after_filtering:>7}"
        )
        print()
        self._update_info(
            "num_datapoints_after_filtering", num_datapoints_after_filtering
        )
        print(
            f"Data extraction steps are finished, data can be found in {self.save_dir} and {self.this_config_save_dir}"
        )
        print()
        return True

    # ==============================================================================
    # Image overlap

    def _overlap_ratio_sufficient(
        self,
        bs1_uv: torch.Tensor,
        bs2_uv: torch.Tensor,
        sonar_datum1: SonarDatum,
        sonar_datum2: SonarDatum,
        num_bs1: int,
        num_bs2: int,
    ) -> bool:
        """
        Find out if the ratio of overlap of one image pair is above the threshold
        """
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
            self.n_elevations,
        )
        bs2_uv_proj = warp_image_batch(
            bs2_uv.unsqueeze(0),
            [sonar_datum2.params],
            [sonar_datum2.pose],
            [sonar_datum1.params],
            [sonar_datum1.pose],
            (width, height),
            self.n_elevations,
        )
        num_bs1_proj = (~bs1_uv_proj.isnan()).sum() / 2
        num_bs2_proj = (~bs2_uv_proj.isnan()).sum() / 2
        ratio1 = num_bs1_proj / num_bs1
        ratio2 = num_bs2_proj / num_bs2
        return float((ratio1.float() + ratio2.float()) / 2) > self.overlap_threshold

    def calculate_overlap(self) -> bool:
        """
        Calculate image overlap by projecting bright spots (pixels above brightness
        threshold) to each other image and only keep tuples that average a ratio above
        the overlap threshold in terms of number of bright spots that appear after
        projection.

        Returns
        -------
        bool
            indicating success of operations

        """
        print("Calculate the image overlap of image pairs")

        if len(self.unfiltered_poses) < 2:
            warn("Not enough sonar data found, skipping the rest of the operations")
            return False

        # Create SonarDatum datapoints
        sonar_datum: list[SonarDatum] = []
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
            blurred_im = blur_image(im)
            blurred_im_tensor = image_to_tensor(blurred_im, self.im_shape)
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
            sonar_datum.append(
                SonarDatum(
                    image=blurred_im_tensor,
                    pose=camera_pose,
                    params=camera_params,
                    stamp=stamp,
                )
            )

        # Find bright spots
        bright_spots: dict[int, torch.Tensor] = {}
        num_bright_spots: dict[int, int] = {}
        for this_sonar_datum in sonar_datum:
            bs_uv = find_bright_spots(
                this_sonar_datum.image, self.conv_size, self.bs_threshold
            )
            bright_spots[this_sonar_datum.stamp] = bs_uv
            num_bright_spots[this_sonar_datum.stamp] = int(
                (~bs_uv.isnan()).sum() * self.n_elevations / 2
            )

        # Check the ratio of overlap of each image pair, keep the ones above the threshold
        tuple_stamps: list[tuple[int, int]] = []
        for sonar_datum1, sonar_datum2 in tqdm(itertools.combinations(sonar_datum, 2)):
            if self._overlap_ratio_sufficient(
                bright_spots[sonar_datum1.stamp],
                bright_spots[sonar_datum2.stamp],
                sonar_datum1,
                sonar_datum2,
                num_bright_spots[sonar_datum1.stamp],
                num_bright_spots[sonar_datum2.stamp],
            ):
                # Randomly choose the order of the tuple
                chosen_tuple = random.choice(
                    [
                        (sonar_datum1.stamp, sonar_datum2.stamp),
                        (sonar_datum2.stamp, sonar_datum1.stamp),
                    ]
                )
                tuple_stamps.append(chosen_tuple)

        num_tuples = len(tuple_stamps)
        num_bs_mean = np.array([num_bs for num_bs in num_bright_spots.values()]).mean()
        self._update_info("num_tuples", num_tuples)
        print(f"Number of tuples with sufficient overlap:   {num_tuples:>7}")
        print(f"Average number of bright spots detected per image:  {num_bs_mean:>7}")
        with open(self.this_config_save_dir / "tuple_stamps.json", "w") as f:
            json.dump(tuple_stamps, f, indent=2)
        return True
