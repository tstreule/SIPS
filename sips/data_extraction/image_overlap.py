import glob
import itertools
import json
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchvision.transforms as T
from matplotlib import pyplot as plt
from PIL import Image, ImageFilter
from torch import nn

from sips.data import CameraParams, CameraPose, SonarDatumStamped
from sips.utils.keypoint_matching import _unravel_index_2d
from sips.utils.point_projection import warp_image, warp_image_batch


class ImageOverlap:
    def __init__(
        self,
        rosbag: str | Path,
        conv_size: int,
        bs_threshold: int,
        n_elevations: int,
        overlap_threshold: float,
    ) -> None:
        self.rosbag = rosbag
        self.source_dir = Path("data/filtered") / Path(rosbag).with_suffix("")
        self.bs_threshold = bs_threshold
        self.n_elevations = n_elevations
        self.overlap_threshold = overlap_threshold
        self.conv_size = conv_size
        self.read_data = False

        self.image_pair_stamps: list[tuple[int, int]] = []
        self.sonar_datum: list[SonarDatumStamped] = []
        self.info: dict[
            str, int | list[str] | dict[str, float | str | None] | dict[str, float]
        ]

    def _find_pose(self, im_stamp, poses) -> CameraPose:
        for this_pose in poses:
            if this_pose["timestamp"] == im_stamp:
                cam_pose = CameraPose(
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
                return cam_pose

        raise ValueError("Image timestamp not found in pose data")

    def _find_sonar_datum(self, im_stamp) -> SonarDatumStamped:
        for sonar_datum in self.sonar_datum:
            if im_stamp == sonar_datum.stamp:
                return sonar_datum
        raise ValueError("Image timestamp not found in self.sonar_datum")

    def _blurred_tensor(self, im: Image.Image) -> torch.Tensor:
        transform = T.Compose([T.PILToTensor()])
        return transform((im.resize((512, 512))).filter(ImageFilter.BLUR))

    # Find blind spots that are brighter than threshold. Only keep one per 8x8
    # grid. Return as uv coordinates
    def _find_bright_spots(self, im: torch.Tensor) -> torch.Tensor:
        maxpool = nn.MaxPool2d(
            self.conv_size, stride=self.conv_size, return_indices=True
        )
        pool, indices = maxpool(im.to(torch.float))  # type: ignore
        mask = pool > self.bs_threshold
        masked_indices = indices[mask]
        row, col = _unravel_index_2d(masked_indices, [4096, 512])
        bright_spots_temp = torch.column_stack((row, col)).to(torch.float)
        bright_spots = torch.full((64 * 64, 2), torch.nan)
        bright_spots[: bright_spots_temp.shape[0], :] = bright_spots_temp
        return bright_spots.permute(1, 0).view(2, 64, 64)

    def _calculate_overlap(
        self,
        bs1_uv: torch.Tensor,
        bs2_uv: torch.Tensor,
        bs1_uv_proj: torch.Tensor,
        bs2_uv_proj: torch.Tensor,
    ) -> float:
        num_bs1_potential = (~bs1_uv.isnan()).sum() * self.n_elevations / 2
        num_bs2_potential = (~bs2_uv.isnan()).sum() * self.n_elevations / 2
        num_bs1_proj = (~bs1_uv_proj.isnan()).sum() / 2
        num_bs2_proj = (~bs2_uv_proj.isnan()).sum() / 2
        ratio1 = num_bs1_proj / num_bs1_potential
        ratio2 = num_bs2_proj / num_bs2_potential
        return float((ratio1.float() + ratio2.float()) / 2)

    def _sufficient_overlap(
        self,
        bs1_uv: torch.Tensor,
        bs2_uv: torch.Tensor,
        sonar_datum1: SonarDatumStamped,
        sonar_datum2: SonarDatumStamped,
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
        ratio = self._calculate_overlap(bs1_uv, bs2_uv, bs1_uv_proj, bs2_uv_proj)
        # _, ax = plt.subplots(2)
        # ax[0].imshow(sonar_datum1.image.reshape(512, 512), cmap="gray")
        # ax[0].plot(bs1_uv[1], bs1_uv[0], ".r", alpha=0.5)
        # ax[1].imshow(sonar_datum2.image.reshape(512, 512), cmap="gray")
        # ax[1].plot(bs2_uv[1], bs2_uv[0], ".r", alpha=0.5)
        # ax[0].plot(
        #     bs2_uv_proj[:, :, 1, ...].flatten(),
        #     bs2_uv_proj[:, :, 0, ...].flatten(),
        #     ".b",
        #     alpha=0.5,
        # )
        # ax[1].plot(
        #     bs1_uv_proj[:, :, 1, ...].flatten(),
        #     bs1_uv_proj[:, :, 0, ...].flatten(),
        #     ".b",
        #     alpha=0.5,
        # )
        # plt.show()
        return ratio > self.overlap_threshold

    def load_data(self) -> None:
        try:
            with open(self.source_dir / "info.json", "r") as f:
                self.info = json.load(f)
        except IOError:
            print("info.json file missing, abort")
            return
        self.camera_params = CameraParams(
            min_range=self.info["sonar_config_params"]["min_range"],  # type: ignore
            max_range=self.info["sonar_config_params"]["max_range"],  # type: ignore
            azimuth=self.info["sonar_config_params"]["horizontal_fov"],  # type: ignore
            elevation=self.info["sonar_config_params"]["vertical_fov"],  # type: ignore
        )
        try:
            with open(self.source_dir / "pose_data.json", "r") as f:
                self.poses = json.load(f)
        except IOError:
            print("pose_data.json file missing, abort")
            return

        for im_path in glob.glob(f"{str(self.source_dir)}/images/*"):
            im_stamp = int(im_path.split("_")[1].split(".")[0])
            im = Image.open(im_path)
            im_pose = self._find_pose(im_stamp, self.poses)
            im_tensor = self._blurred_tensor(im)
            self.sonar_datum.append(
                SonarDatumStamped(
                    image=im_tensor,
                    pose=im_pose,
                    params=self.camera_params,
                    stamp=im_stamp,
                )
            )
        if len(self.sonar_datum) < 2:
            print("not enough sonar data found, abort")
            return
        self.read_data = True
        return

    def find_pairs(self) -> None:
        bright_spots: dict[int, torch.Tensor] = {}
        for sonar_datum in self.sonar_datum:
            bs_uv = self._find_bright_spots(sonar_datum.image)
            bright_spots[sonar_datum.stamp] = bs_uv
        for sonar_datum1, sonar_datum2 in itertools.combinations(self.sonar_datum, 2):
            if self._sufficient_overlap(
                bright_spots[sonar_datum1.stamp],
                bright_spots[sonar_datum2.stamp],
                sonar_datum1,
                sonar_datum2,
            ):
                self.image_pair_stamps.append((sonar_datum1.stamp, sonar_datum2.stamp))
                self.image_pair_stamps.append((sonar_datum2.stamp, sonar_datum1.stamp))

        return

    def plot(self) -> None:
        if len(self.image_pair_stamps) < 1:
            print(
                "Plotting not possible as there were no overlapping images found, returning"
            )
            return
        idx = random.randrange(len(self.image_pair_stamps))
        stamp1, stamp2 = self.image_pair_stamps[idx]
        sonar_datum1 = self._find_sonar_datum(stamp1)
        sonar_datum2 = self._find_sonar_datum(stamp2)
        bs1_uv = self._find_bright_spots(sonar_datum1.image)
        bs2_uv = self._find_bright_spots(sonar_datum2.image)
        height, width = sonar_datum1.image.shape[1:]
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
        _, ax = plt.subplots(2, 2)
        ax[0, 0].imshow(sonar_datum1.image.reshape(512, 512), cmap="gray")
        ax[0, 1].imshow(sonar_datum2.image.reshape(512, 512), cmap="gray")
        ax[1, 0].imshow(sonar_datum1.image.reshape(512, 512), cmap="gray")
        ax[1, 0].plot(bs1_uv[1], bs1_uv[0], ".r", alpha=0.5)
        ax[1, 1].imshow(sonar_datum2.image.reshape(512, 512), cmap="gray")
        ax[1, 1].plot(bs2_uv[1], bs2_uv[0], ".r", alpha=0.5)
        ax[1, 0].plot(
            bs2_uv_proj[:, :, 1, ...].flatten(),
            bs2_uv_proj[:, :, 0, ...].flatten(),
            ".b",
            alpha=0.5,
        )
        ax[1, 1].plot(
            bs1_uv_proj[:, :, 1, ...].flatten(),
            bs1_uv_proj[:, :, 0, ...].flatten(),
            ".b",
            alpha=0.5,
        )
        plt.show()
        print()

    def finish(self) -> None:
        self.info["num_tuples"] = len(self.image_pair_stamps)
        with open(self.source_dir / "info.json", "w") as f:
            json.dump(self.info, f, indent=2)
        with open(self.source_dir / "tuple_stamps.json", "w") as f:
            json.dump(self.image_pair_stamps, f, indent=2)


def main(
    conv_size: int = 8,
    bs_threshold: int = 10,
    n_elevations: int = 5,
    overlap_threshold: float = 1,
    plot: bool = True,
    seed: Optional[int] = 240,
):
    random.seed(seed)
    rosbags = [
        "freeRoaming15SonarFels2-0805.bag",
    ]

    for this_rosbag in rosbags:
        print(100 * "=")
        print(f"Start processing {this_rosbag}")
        image_overlap = ImageOverlap(
            this_rosbag,
            conv_size,
            bs_threshold,
            n_elevations,
            overlap_threshold,
        )
        image_overlap.load_data()
        if image_overlap.read_data:
            image_overlap.find_pairs()
            print(len(image_overlap.image_pair_stamps))
            if plot:
                image_overlap.plot()
            image_overlap.finish()
            print()
    return


if __name__ == "__main__":
    main()
