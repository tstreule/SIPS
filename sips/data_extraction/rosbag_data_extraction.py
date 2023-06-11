import json
from pathlib import Path

import cv2  # type: ignore
import numpy as np
import numpy.typing as npt
from PIL import Image
from rosbags.rosbag1 import Reader
from rosbags.serde import deserialize_cdr, ros1_to_cdr
from rosbags.typesys import get_types_from_msg, register_types
from rosbags.typesys.types import sensor_msgs__msg__Image as ImageMessage
from scipy.interpolate import interp1d
from termcolor import cprint

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

    def save_img_if_okay(self, message: ImageMessage, save_path: str | Path) -> None:
        keyframe = message.data.reshape((message.height, message.width))

        # Skip if image is redundant
        if self._is_keyframe_redundant(keyframe):
            self.n_redundant += 1
            return

        # Save image
        self.n_saved += 1
        Image.fromarray(keyframe, mode="L").save(save_path)

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


# ==============================================================================
# Pose and twist data filtering
# Idea: filter in accordance to movement script probably not possible as the robot
# drifts and will not be crazy accurate


def filter_pose_data(pose_data):
    print("WARNING: Filtering POSE data not implemented.")
    return pose_data


def filter_twist_data(twist_data):
    print("WARNING: Filtering TWIST data not implemented.")
    return twist_data


class DataExtraction:
    """
    Extract, match and filter the data
    """

    def __init__(self, rosbag: str | Path) -> None:
        self.rosbag = rosbag
        self.rosbag_path = Path("data/raw") / rosbag
        self.save_dir = Path("data/filtered") / Path(rosbag).with_suffix("")
        self.rosbag_connections: list[str] = []
        self.image_filter_params: dict[str, float | str | None] = {}
        self.poses: list[dict[str, int | dict[str, float] | list[float]]] = []
        self.sonar_images: dict[int, ImageMessage] = {}
        self.sonar_params: dict[str, float] = {}

        self.extracted_sonar = 0
        self.extracted_pose = 0
        self.matched_datapoints = 0
        self.datapoints_after_filtering = 0

        self.extracted_data = False

        with open("scripts/bag-data-extraction/misc/SonarConfiguration.msg", "r") as f:
            SONAR_CONFIGURATION_MSG = f.read()
        with open(
            "scripts/bag-data-extraction/misc/SonarConfigurationChange.msg", "r"
        ) as f:
            SONAR_CONFIGURATION_CHANGE_MSG = f.read()

        register_types(
            get_types_from_msg(
                SONAR_CONFIGURATION_MSG,
                "sonar_driver_interfaces/msg/SonarConfiguration",
            )
        )
        register_types(
            get_types_from_msg(
                SONAR_CONFIGURATION_CHANGE_MSG,
                "sonar_driver_interfaces/msg/SonarConfigurationChange",
            )
        )
        from rosbags.typesys.types import (  # type: ignore  # noqa
            sonar_driver_interfaces__msg__SonarConfiguration as SonarConfiguration,
        )
        from rosbags.typesys.types import (  # type: ignore  # noqa
            sonar_driver_interfaces__msg__SonarConfigurationChange as SonarConfigurationChange,
        )

    # TODO: check if pose reccording frequency and other recording parameters are correct
    def check_connections(self) -> None:
        with Reader(self.rosbag_path) as reader:
            for connection in reader.connections:
                self.rosbag_connections.append(
                    f"{connection.topic} {connection.msgtype}"
                )

    def extract_data(self) -> None:
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
                cprint(
                    "No pose data or sonar data found in this bag, skipping the rest of the operations",
                    "red",
                )
                print()
                return
            sonar_config_connections = [
                x for x in reader.connections if x.topic == sonar_config_topic
            ]
            if len(sonar_config_connections) < 1:
                cprint(
                    "No Sonar configuration data found in this bag, skipping the rest of the operations",
                    "red",
                )
                print()
                return

            connections = [
                x
                for x in reader.connections
                if x.topic in [sonar_topic, pose_topic, sonar_config_topic]
            ]

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

                if (
                    connection.msgtype
                    == "sonar_driver_interfaces/msg/SonarConfiguration"
                ):
                    self.sonar_params = {
                        "horizontal_fov": msg.horz_fov,
                        "vertical_fov": msg.vert_fov,
                        "max_range": msg.max_range,
                        "min_range": msg.min_range,
                    }
                    if len(self.sonar_params) > 0:
                        if (
                            self.sonar_params["horizontal_fov"] != msg.horz_fov
                            or self.sonar_params["vertical_fov"] != msg.vert_fov
                            or self.sonar_params["max_range"] != msg.max_range
                            or self.sonar_params["min_range"] != msg.min_range
                        ):
                            cprint("real change", "red")

                if (
                    connection.msgtype
                    == "sonar_driver_interfaces/msg/SonarConfiguration"
                ):
                    if (
                        self.sonar_params["horizontal_fov"] != msg.horz_fov
                        or self.sonar_params["vertical_fov"] != msg.vert_fov
                        or self.sonar_params["max_range"] != msg.max_range
                        or self.sonar_params["min_range"] != msg.min_range
                    ):
                        cprint("real change", "red")
                        print()
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
                cprint(
                    f"Frequency of pose data too high ({pose_frequency:.2f}Hz)"
                    ", skipping the rest of the operations",
                    "red",
                )
                print()
                return
            self.extracted_sonar = len(self.sonar_images)
            self.extracted_pose = len(self.poses)
            print(f"Number of extracted sonar images:   {self.extracted_sonar:>7}")
            print(f"Number of extracted pose data:   {self.extracted_pose:>7}")
            print()
            self.extracted_data = True

    def match_data(self) -> None:
        print("Match sonar data with poses through interpolation")

        # Read the sonar timestamps in ascending order
        sonar_stamps = sorted(list(self.sonar_images.keys()))

        # First sonar datapoint should always happen after first pose datapoint
        # to improve accuracy of interpolation. The last sonar datapoint should
        # always happen before the last pose datapoint for the same reason
        assert (
            self.poses[0]["timestamp"] < sonar_stamps[0]  # type: ignore
        ), f"{30 * '='}\nFirst sonar datapoint before first pose datapoint\n{30 * '='}"  # type: ignore
        if self.poses[-1]["timestamp"] < sonar_stamps[-1]:  # type: ignore
            print(
                "Last sonar datapoint after last pose datapoint, removing sonar data after last pose datapoint"
            )
            for i in range(2, len(self.poses)):
                if sonar_stamps[-i] <= self.poses[-1]["timestamp"]:  # type: ignore
                    sonar_stamps_matched = sonar_stamps[: -i + 1]
                    sonar_stamps_unmatched = sonar_stamps[-i + 1 :]
                    break
        else:
            sonar_stamps_matched = sonar_stamps
            sonar_stamps_unmatched = []
        assert (
            self.poses[-1]["timestamp"] >= sonar_stamps_matched[-1]  # type: ignore
        ), f"{30 * '='}\nLast sonar datapoint after last pose datapoint\n{30 * '='}"

        # Remove the sonar images that were recorded after the last pose
        # datapoint
        for unmatch in sonar_stamps_unmatched:
            self.sonar_images.pop(unmatch)

        # Create interpolation over all pose and twist datapoints
        # then find interpolated pose and twist datapoint for each
        # sonar timestamp
        self.poses = match_pose_data(sonar_stamps_matched, self.poses)
        assert len(self.poses) == len(
            self.sonar_images
        ), "length of matched data does not match"
        self.matched_datapoints = len(self.sonar_images)
        print(f"Number of matched datapoints:   {self.matched_datapoints:>7}")
        print()

    def filter_data(self) -> None:
        print("Filter sonar images with redudant image filter")

        save_dir = Path("data/filtered")
        save_dir.mkdir(exist_ok=True)
        save_dir = save_dir.joinpath(Path(self.rosbag).with_suffix(""))
        save_dir.mkdir(exist_ok=True)
        (save_dir / "images").mkdir(exist_ok=True)

        image_filter = RedundantImageFilter(blur="bilateral")
        self.image_filter_params = {
            "blur": image_filter.blur,
            "blur_size": image_filter.blur_size,
            "blur_std": image_filter.blur_std,
            "threshold": image_filter.threshold,
        }
        unfiltered_poses: list[dict[str, int | dict[str, float] | list[float]]] = []
        n_redundant = 0
        sonar_timestamps = sorted(list(self.sonar_images.keys()))
        for pose, sonar_timestamp in zip(self.poses, sonar_timestamps):
            assert pose["timestamp"] == sonar_timestamp
            fname = f"sonar_{sonar_timestamp}.png"
            img = self.sonar_images[sonar_timestamp]
            image_filter.save_img_if_okay(img, save_dir / "images" / fname)
            if image_filter.n_redundant == n_redundant:
                unfiltered_poses.append(pose)
            n_redundant = image_filter.n_redundant
        self.poses = unfiltered_poses
        assert len(self.poses) == image_filter.n_saved
        self.datapoints_after_filtering = len(self.poses)
        with open(save_dir / "pose_data.json", "w") as fp:
            json.dump(self.poses, fp, indent=2)
        print(
            f"Number of datapoints after filtering:   {self.datapoints_after_filtering:>7}"
        )
        print()
        print(f"Data extraction steps are finished, data can be found in {save_dir}")
        print()

    def finish(self) -> None:
        save_dir = Path("data/filtered")
        save_dir.mkdir(exist_ok=True)
        save_dir = save_dir.joinpath(Path(self.rosbag).with_suffix(""))
        save_dir.mkdir(exist_ok=True)
        (save_dir / "images").mkdir(exist_ok=True)
        info: dict[
            str, int | list[str] | dict[str, float | str | None] | dict[str, float]
        ] = {
            "extracted_sonar_datapoints": self.extracted_sonar,
            "extracted_pose_datapoints": self.extracted_pose,
            "matched_datapoints": self.matched_datapoints,
            "datapoints_after_filtering": self.datapoints_after_filtering,
            "image_filter_params": self.image_filter_params,
            "sonar_config_params": self.sonar_params,
            "rosbag_connections_and_topics": self.rosbag_connections,
        }
        with open(self.save_dir / "info.json", "w") as fp:
            json.dump(info, fp, indent=2)


# ==============================================================================
# Interpolators for both sonar and twist data


def make_pose_interpolator(poses: list[dict[str, int | dict[str, float]]]) -> interp1d:
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


# ==============================================================================
# Match interpolated pose and twist data with sonar timestamps


def match_pose_data(sonar_stamps, poses):
    pose_interp = make_pose_interpolator(poses)
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
# MAIN


# TODO: think about how to decide what rosbags we wanna take
# Probably does not make sense to just process all data that
# has not been extracted, as we might not be interested in the
# data of each and every bag. For now it's annoying to provide
# the bags at runtime but probably best approach in later stages
# with rosnode and stuff.
def main():
    rosbags = [
        "freeRoaming15SonarFels2-0805.bag",
        "freeRoaming15SonarFeSteg-0805.bag",
        "freeRoaming15-0805.bag",
        "freeRoaming15SonarFels-0805.bag",
        "freeRoaming45deg-0805.bag",
        "freeRoaming1-0805.bag",
        "freeRoaming15Sonar2-0805.bag",
        "freeRoaming15Sonar-0805.bag",
        "freeRoaming15SonarAuto-0805.bag",
    ]

    for this_rosbag in rosbags:
        print(100 * "=")
        print(f"Start processing {this_rosbag}")
        data_extractor = DataExtraction(this_rosbag)
        data_extractor.check_connections()
        data_extractor.extract_data()
        if data_extractor.extracted_data:
            data_extractor.match_data()
            data_extractor.filter_data()
        data_extractor.finish()


if __name__ == "__main__":
    main()
