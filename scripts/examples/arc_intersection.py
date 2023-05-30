"""
Example script to time and visualize arc projections and matching of sonar images.

"""
from time import time

import torch
from matplotlib import pyplot as plt

from scripts.examples._sonar_data import get_random_batch
from sips.utils.keypoint_matching import batch_match_keypoints_2d
from sips.utils.plotting import (
    COLOR_HIGHLIGHT,
    COLOR_SONAR_1,
    COLOR_SONAR_2,
    plot_arcs_2d,
    plot_arcs_3d,
)
from sips.utils.point_projection import batch_uv_to_xyz, batch_warp_image


def make_uv_grid(
    width: int, height: int, conv_size: int, batch_size: int = 1, *, add_noise=True
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Creates a (2D) image grid, simulating sonar keypoints.

    """
    # Make grid
    xs = torch.arange(0, width, conv_size) + conv_size / 2
    ys = torch.arange(0, height, conv_size) + conv_size / 2
    mesh = torch.meshgrid(xs, ys, indexing="ij")
    grid = torch.stack(batch_size * [torch.stack(mesh)]).float()
    # Define (random) keypoints for both images
    alpha = 0.7
    torch.random.manual_seed(420)
    std = torch.broadcast_to(torch.tensor(alpha * conv_size), grid.shape)
    if not add_noise:
        return grid, grid
    return grid + torch.normal(0, std), grid + torch.normal(0, std)


def main(
    batch_size: int = 8,
    conv_size: int = 8,
    distance_threshold: float = 0.5,
    plot: bool = False,
    timeit: bool = True,
):

    # Set data
    batch = get_random_batch(batch_size=batch_size)
    _, _, width, height = batch.image1.shape

    # Create convolutional grid
    kp1_uv, kp2_uv = make_uv_grid(width, height, conv_size, batch_size, add_noise=True)

    # Get arc position
    kp1_xyz = batch_uv_to_xyz(kp1_uv, batch.params1, batch.pose1, (width, height))
    kp2_xyz = batch_uv_to_xyz(kp2_uv, batch.params2, batch.pose2, (width, height))

    # Point projection to other camera
    # kp2_uv_proj = batch_xyz_to_uv(kp2_xyz, batch.params1, batch.pose1, (width, height))
    kp2_uv_proj = batch_warp_image(
        kp2_uv, batch.params2, batch.pose2, batch.params1, batch.pose1, (width, height)
    )

    # Find matching keypoints that are close
    matches_uv = batch_match_keypoints_2d(
        kp1_uv, kp2_uv_proj, conv_size, distance_threshold
    )

    # Optional: Plot point projection (select first in batch)
    if plot:
        _, D, _, W, H = kp1_xyz.shape
        kp1_xyz_0 = kp1_xyz[0].view(D, 3, W * H).permute(2, 0, 1)
        kp2_xyz_0 = kp2_xyz[0].view(D, 3, W * H).permute(2, 0, 1)
        kp1_uv_0 = kp1_uv[0].view(2, W * H).permute(1, 0)
        kp2_uv_proj_0 = kp2_uv_proj[0].view(D, 2, W * H).permute(2, 0, 1)

        nan_mask = torch.isnan(kp2_uv_proj_0).any(-1)
        kp2_xyz_filtered = kp2_xyz_0.clone()
        kp2_xyz_filtered[nan_mask] = torch.nan
        plot_arcs_3d(
            [kp1_xyz_0[::12], kp2_xyz_0[::12], kp2_xyz_filtered[::12]],
            [batch.pose1[0].position, batch.pose2[0].position, None],
            colors=[COLOR_SONAR_1, COLOR_SONAR_2, COLOR_HIGHLIGHT],
        )
        plot_arcs_2d(
            [kp1_uv_0, kp2_uv_proj_0],
            (width, height),
            conv_size,
            [COLOR_SONAR_1, COLOR_SONAR_2],
            (matches_uv[0][0].cpu(), matches_uv[0][1].cpu()),
            COLOR_HIGHLIGHT,
        )
        plt.show()

    # Optional: Measure how long it takes
    if timeit:
        times = []
        for _ in range(100):
            # fmt: off
            start = time()
            kp2_uv_proj = batch_warp_image(kp2_uv, batch.params2, batch.pose2, batch.params1, batch.pose1, (width, height))
            batch_match_keypoints_2d(kp1_uv, kp2_uv_proj, conv_size, distance_threshold)
            times.append(time() - start)
            # fmt: on
        print("Mean time:", torch.tensor(times).mean())
        print("Mean time per batch:", torch.tensor(times).mean() / batch_size)

    return None


if __name__ == "__main__":
    main()
