import json
import random
from json import JSONDecodeError
from pathlib import Path
from warnings import warn

import torch
import torchvision.transforms as T
from matplotlib import pyplot as plt
from PIL import Image

from sips.data import CameraParams, CameraPose, SonarDatum
from sips.data_extraction.utils.handle_configs import find_config_num
from sips.data_extraction.utils.image_operations import (
    find_bright_spots,
    image_to_tensor,
)
from sips.utils.point_projection import warp_image, warp_image_batch  # TODO: choose one


def _pose_empty(pose: CameraPose) -> bool:
    is_empty_position = bool((pose.position == torch.tensor((-1, -1, -1))).all())
    is_empty_rotation = bool(
        (pose.rotation == torch.tensor((-0.5, -0.5, -0.5, -0.5))).all()
    )
    return is_empty_position and is_empty_rotation


def plot_overlap(config, rosbag) -> None:
    random.seed(config.seed)
    source_dir = Path("data/filtered") / Path(rosbag).with_suffix("")
    image_dir = source_dir / "images"
    config_num = find_config_num(config, source_dir)
    if config_num == -1:
        warn("Unable to find valid config number")
        return
    source_dir = source_dir / str(config_num)
    try:
        with open(source_dir / "tuple_stamps.json") as f:
            tuple_stamps = json.load(f)
    except IOError:
        warn(f"tuple_stamps.json file")
        return
    except JSONDecodeError:
        warn(f"tuple_stamps.json file empty")
        return
    idx = random.randrange(len(tuple_stamps))
    stamp0, stamp1 = tuple_stamps[idx]
    im_path0 = image_dir / f"sonar_{stamp0}.png"
    im_path1 = image_dir / f"sonar_{stamp1}.png"
    try:
        im0 = Image.open(im_path0)
    except FileNotFoundError:
        warn("Image file not found, skipping the rest of the operations")
        return
    im_tensor0 = image_to_tensor(im0)
    try:
        im1 = Image.open(im_path1)
    except FileNotFoundError:
        warn("Image file not found, skipping the rest of the operations")
        return
    im_tensor1 = image_to_tensor(im1)
    with open(source_dir / "unfiltered_pose_data.json") as f:
        poses = json.load(f)
    pose0 = CameraPose(position=[-1, -1, -1], rotation=[-1, -1, -1, -1])
    pose1 = CameraPose(position=[-1, -1, -1], rotation=[-1, -1, -1, -1])
    for this_pose in poses:
        if (not _pose_empty(pose0)) and (not _pose_empty(pose1)):
            break  # Both poses have been found, no need to continue loop
        if this_pose["timestamp"] == stamp0:
            pose0 = CameraPose(
                position=[
                    this_pose["point_position"]["x"],
                    this_pose["point_position"]["y"],
                    this_pose["point_position"]["z"],
                ],
                rotation=[
                    this_pose["quaterion_orientation"]["x"],
                    this_pose["quaterion_orientation"]["y"],
                    this_pose["quaterion_orientation"]["z"],
                    this_pose["quaterion_orientation"]["w"],
                ],
            )
            continue
        if this_pose["timestamp"] == stamp1:
            pose1 = CameraPose(
                position=[
                    this_pose["point_position"]["x"],
                    this_pose["point_position"]["y"],
                    this_pose["point_position"]["z"],
                ],
                rotation=[
                    this_pose["quaterion_orientation"]["x"],
                    this_pose["quaterion_orientation"]["y"],
                    this_pose["quaterion_orientation"]["z"],
                    this_pose["quaterion_orientation"]["w"],
                ],
            )
            continue
    if _pose_empty(pose0) or _pose_empty(pose1):
        warn("Pose data not found, skipping the rest of the operations")
        return
    params = CameraParams(
        min_range=config.min_range,
        max_range=config.max_range,
        azimuth=config.horizontal_fov,
        elevation=config.vertical_fov,
    )

    sonar_datum0 = SonarDatum(
        image=im_tensor0,
        pose=pose0,
        params=params,
    )
    sonar_datum1 = SonarDatum(
        image=im_tensor1,
        pose=pose1,
        params=params,
    )
    bs0_uv = find_bright_spots(
        sonar_datum0.image, config.conv_size, config.bright_spots_threshold
    )
    bs1_uv = find_bright_spots(
        sonar_datum1.image, config.conv_size, config.bright_spots_threshold
    )
    height, width = sonar_datum1.image.shape[1:]
    bs0_uv_proj = warp_image_batch(
        bs0_uv.unsqueeze(0),
        [sonar_datum0.params],
        [sonar_datum0.pose],
        [sonar_datum1.params],
        [sonar_datum1.pose],
        (width, height),
        config.n_elevations,
    )
    bs1_uv_proj = warp_image_batch(
        bs1_uv.unsqueeze(0),
        [sonar_datum1.params],
        [sonar_datum1.pose],
        [sonar_datum0.params],
        [sonar_datum0.pose],
        (width, height),
        config.n_elevations,
    )

    black = torch.full((512, 512), 0)
    _, ax = plt.subplots(2, 3)
    ax[0, 0].imshow(sonar_datum0.image.reshape(512, 512), cmap="gray")
    ax[0, 1].imshow(black, cmap="gray")
    ax[0, 1].plot(bs0_uv[1], bs0_uv[0], ",w")
    ax[0, 2].imshow(sonar_datum0.image.reshape(512, 512), cmap="gray")
    ax[0, 2].plot(bs0_uv[1], bs0_uv[0], ".r", alpha=0.5)
    ax[0, 2].plot(
        bs1_uv_proj[:, :, 1, ...].flatten(),
        bs1_uv_proj[:, :, 0, ...].flatten(),
        ".b",
        alpha=0.5,
    )
    ax[1, 0].imshow(sonar_datum1.image.reshape(512, 512), cmap="gray")
    ax[1, 1].imshow(black, cmap="gray")
    ax[1, 1].plot(bs1_uv[1], bs1_uv[0], ",w")
    ax[1, 2].imshow(sonar_datum1.image.reshape(512, 512), cmap="gray")
    ax[1, 2].plot(bs1_uv[1], bs1_uv[0], ".r", alpha=0.5)
    ax[1, 2].plot(
        bs0_uv_proj[:, :, 1, ...].flatten(),
        bs0_uv_proj[:, :, 0, ...].flatten(),
        ".b",
        alpha=0.5,
    )
    plt.show()
    return
    _, ax = plt.subplots(2, 2)
    ax[0, 0].imshow(sonar_datum0.image.reshape(512, 512), cmap="gray")
    ax[0, 1].imshow(sonar_datum1.image.reshape(512, 512), cmap="gray")
    ax[1, 0].imshow(sonar_datum0.image.reshape(512, 512), cmap="gray")
    ax[1, 0].plot(bs0_uv[1], bs0_uv[0], ".r", alpha=0.5)
    ax[1, 1].imshow(sonar_datum1.image.reshape(512, 512), cmap="gray")
    ax[1, 1].plot(bs1_uv[1], bs1_uv[0], ".r", alpha=0.5)
    ax[1, 0].plot(
        bs1_uv_proj[:, :, 1, ...].flatten(),
        bs1_uv_proj[:, :, 0, ...].flatten(),
        ".b",
        alpha=0.5,
    )
    ax[1, 1].plot(
        bs0_uv_proj[:, :, 1, ...].flatten(),
        bs0_uv_proj[:, :, 0, ...].flatten(),
        ".b",
        alpha=0.5,
    )
    plt.show()
    return
