import json
import os
from json import JSONDecodeError
from pathlib import Path
from warnings import warn

import cv2  # type: ignore
import numpy as np
import numpy.typing as npt
from PIL import Image
from rosbags.rosbag1 import Reader
from rosbags.serde import deserialize_cdr, ros1_to_cdr
from rosbags.typesys import get_types_from_msg, register_types
from rosbags.typesys.types import sensor_msgs__msg__Image as ImageMessage
from scipy.interpolate import interp1d

from sips.configs.base_config import _DatasetsConfig
from sips.data_extraction.utils.handle_configs import add_entry_to_summary

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


class DataExtraction:
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

        self.poses: list[dict[str, int | dict[str, float] | list[float]]] = []
        self.sonar_images: dict[int, ImageMessage] = {}

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
        self.save_dir = self.save_dir / str(config_num)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        if not os.path.isfile(self.save_dir / "info.json"):
            with open(self.save_dir / "info.json", "w") as f:
                pass

    def _update_info(
        self,
        key: str,
        value: int | list[str] | dict[str, float | str | None] | dict[str, float],
    ) -> None:
        with open(self.save_dir / "info.json", "r") as f:
            try:
                info = json.load(f)
            except JSONDecodeError:
                info = {}
        with open(self.save_dir / "info.json", "w") as f:
            info[key] = value
            json.dump(info, f, indent=2)
        return

    # ==============================================================================
    # Match interpolated pose data with sonar timestamps

    def _match_pose_data(self, sonar_stamps, poses):
        pose_interp = self._make_pose_interpolator(poses)
        pose_data_matched = []
        for sonar_stamp in sonar_stamps:
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
        with Reader(self.rosbag_path) as reader:
            for connection in reader.connections:
                if connection.topic == sonar_topic:
                    sonar_topic_available = True
                elif connection.topic == pose_topic:
                    pose_topic_available = True
                rosbag_connections.append(f"{connection.topic} {connection.msgtype}")
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
        with Reader(self.rosbag_path) as reader:
            # --------

            sonar_topic = "/proteus_PRA01/sonar_0/image"
            pose_topic = "/proteus_PRA01/gtsam/pose"
            sonar_config_topic = "/proteus_PRA01/sonar_0/configuration"
            sonar_connections = [
                x for x in reader.connections if x.topic == sonar_topic
            ]
            pose_connections = [x for x in reader.connections if x.topic == pose_topic]
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

                if connection.msgtype == "sensor_msgs/msg/Image" and first_pose_seen:
                    timestamp = int(
                        f"{msg.header.stamp.sec}{msg.header.stamp.nanosec:09}"
                    )
                    self.sonar_images[timestamp] = msg

                if connection.msgtype == "geometry_msgs/msg/PoseWithCovarianceStamped":
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
                self._update_info("num_extracted_sonar_datapoints", num_extracted_sonar)
                self._update_info("num_extracted_pose_datapoints", num_extracted_pose)
                return False
            print(f"Number of extracted sonar images:   {num_extracted_sonar:>7}")
            print(f"Number of extracted pose data:   {num_extracted_pose:>7}")
            print()
            self._update_info("num_extracted_sonar_datapoints", num_extracted_sonar)
            self._update_info("num_extracted_pose_datapoints", num_extracted_pose)
            self.num_extracted_sonar = num_extracted_sonar
            self.num_extracted_pose = num_extracted_pose
            return True

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

        unfiltered_poses: list[dict[str, int | dict[str, float] | list[float]]] = []
        sonar_timestamps = sorted(list(self.sonar_images.keys()))
        for pose, sonar_timestamp in zip(self.poses, sonar_timestamps):
            assert (
                pose["timestamp"] == sonar_timestamp
            ), "pose timestamp and sonar timestamp not agreeing"
            fname = f"sonar_{sonar_timestamp}.png"
            img = self.sonar_images[sonar_timestamp]
            image_redundant = image_filter.image_redundant(
                img, self.image_save_dir / fname, save_ims
            )
            if not image_redundant:
                unfiltered_poses.append(pose)
        self.poses = unfiltered_poses
        assert len(self.poses) == image_filter.n_saved
        num_datapoints_after_filtering = len(self.poses)
        if num_datapoints_after_filtering <= 0:
            warn(
                "No remaining sonar data found after applying redundant image filter, skipping the rest of the operations"
            )
            self._update_info(
                "num_datapoints_after_filtering", num_datapoints_after_filtering
            )
            return False
        with open(self.save_dir / "pose_data.json", "w") as fp:
            json.dump(self.poses, fp, indent=2)
        print(
            f"Number of datapoints after filtering:   {num_datapoints_after_filtering:>7}"
        )
        print()
        self._update_info(
            "num_datapoints_after_filtering", num_datapoints_after_filtering
        )
        self.num_datapoints_after_filtering = num_datapoints_after_filtering
        print(
            f"Data extraction steps are finished, data can be found in {self.save_dir}"
        )
        print()
        return True

    # def save_data(self) -> bool:
    #     save_dir = Path("data/filtered")
    #     save_dir.mkdir(exist_ok=True)
    #     save_dir = save_dir.joinpath(Path(self.rosbag).with_suffix(""))
    #     save_dir.mkdir(exist_ok=True)
    #     info: dict[
    #         str, int | list[str] | dict[str, float | str | None] | dict[str, float]
    #     ] = {
    #         "num_extracted_sonar_datapoints": self.num_extracted_sonar,
    #         "num_extracted_pose_datapoints": self.num_extracted_pose,
    #         "num_matched_datapoints": self.num_matched_datapoints,
    #         "num_datapoints_after_filtering": self.num_datapoints_after_filtering,
    #         "image_filter_params": self.image_filter_params,
    #         "sonar_config_params": self.sonar_params,
    #         "rosbag_connections_and_topics": self.rosbag_connections,
    #     }
    #     with open(self.save_dir / "info.json", "w") as fp:
    #         json.dump(info, fp, indent=2)
    #     return True


# ==============================================================================
# MAIN


# # TODO: think about how to decide what rosbags we wanna take
# # Probably does not make sense to just process all data that
# # has not been extracted, as we might not be interested in the
# # data of each and every bag. For now it's annoying to provide
# # the bags at runtime but probably best approach in later stages
# # with rosnode and stuff.
# def main():
#     rosbags = [
#         "freeRoaming15SonarFels2-0805.bag",
#         "freeRoaming15SonarFeSteg-0805.bag",
#         "freeRoaming15-0805.bag",
#         "freeRoaming15SonarFels-0805.bag",
#         "freeRoaming45deg-0805.bag",
#         "freeRoaming1-0805.bag",
#         "freeRoaming15Sonar2-0805.bag",
#         "freeRoaming15Sonar-0805.bag",
#         "freeRoaming15SonarAuto-0805.bag",
#     ]

#     for this_rosbag in rosbags:
#         print(100 * "=")
#         print(f"Start processing {this_rosbag}")
#         data_extractor = DataExtraction(this_rosbag)
#         _ = conns_checked = data_extractor.check_connections()
#         data_extracted = data_extractor.extract_data()
#         if data_extracted:
#             _ = data_extractor.match_data()
#             _ = data_extractor.filter_data()
#         _ = data_extractor.save_data()


# if __name__ == "__main__":
#     main()
