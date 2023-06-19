"""
Example script to time and visualize arc projections and matching of sonar images.

"""
from time import time
from typing import Optional

import torch
import torch.backends.mps
import typer
from matplotlib import pyplot as plt

from scripts.examples._sonar_data import get_random_batch
from sips.utils.keypoint_matching import match_keypoints_2d_batch
from sips.utils.plotting import (
    COLOR_HIGHLIGHT,
    COLOR_SONAR_1,
    COLOR_SONAR_2,
    plot_arcs_2d,
    plot_arcs_3d,
)
from sips.utils.point_projection import uv_to_xyz_batch, warp_image_batch

app = typer.Typer()


def _set_seed(seed: int | None) -> None:
    """
    Set seeds for cpu and cuda/mps if available.

    """
    if seed is None:
        return

    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        from torch import mps  # type: ignore[attr-defined]

        mps.manual_seed(seed)


def _make_uv_grid(
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


@app.command()
def main(
    batch_size: int = 8,
    conv_size: int = 8,
    distance_threshold: float = 0.5,
    plot: bool = True,
    timeit: bool = True,
    metric: str = "p2l",
    device: str = "cpu",
    seed: Optional[int] = 120,
):
    _set_seed(seed)

    # Set data
    batch = get_random_batch(batch_size=batch_size)
    _, _, width, height = batch.image1.shape

    # Create convolutional grid
    kp1_uv, kp2_uv = _make_uv_grid(width, height, conv_size, batch_size, add_noise=True)
    kp1_uv = kp1_uv.to(device)
    kp2_uv = kp2_uv.to(device)

    # Get arc position
    kp1_xyz = uv_to_xyz_batch(kp1_uv, batch.params1, batch.pose1, (width, height))
    kp2_xyz = uv_to_xyz_batch(kp2_uv, batch.params2, batch.pose2, (width, height))

    # Point projection to other camera
    # kp2_uv_proj = batch_xyz_to_uv(kp2_xyz, batch.params1, batch.pose1, (width, height))
    kp2_uv_proj = warp_image_batch(
        kp2_uv, batch.params2, batch.pose2, batch.params1, batch.pose1, (width, height)
    )

    # Find matching keypoints that are close
    kp1_match, kp2_match, _, _, mask = match_keypoints_2d_batch(
        kp1_uv, kp2_uv_proj, conv_size, distance_threshold, distance=metric  # type: ignore[arg-type]
    )

    # Optional: Plot point projection (select first in batch)
    if plot:
        b = 0  # select batch index (b in B)

        _, D, _, W, H = kp1_xyz.shape
        kp1_xyz_b = kp1_xyz[b].view(D, 3, W * H).permute(2, 0, 1)
        kp2_xyz_b = kp2_xyz[b].view(D, 3, W * H).permute(2, 0, 1)
        kp1_uv_b = kp1_uv[b].view(2, W * H).permute(1, 0)
        kp2_uv_proj_b = kp2_uv_proj[b].view(D, 2, W * H).permute(2, 0, 1)

        nan_mask = torch.isnan(kp2_uv_proj_b).any(-1)
        kp2_xyz_filtered = kp2_xyz_b.clone()
        kp2_xyz_filtered[nan_mask] = torch.nan
        plot_arcs_3d(
            [kp1_xyz_b[::12], kp2_xyz_b[::12], kp2_xyz_filtered[::12]],
            [batch.pose1[b].position, batch.pose2[b].position, None],
            colors=[COLOR_SONAR_1, COLOR_SONAR_2, COLOR_HIGHLIGHT],
        )
        plot_arcs_2d(
            [kp1_uv_b, kp2_uv_proj_b],
            (width, height),
            conv_size,
            [COLOR_SONAR_1, COLOR_SONAR_2],
            (kp1_match[b][mask[b]].cpu(), kp2_match[b][mask[b]]),
            COLOR_HIGHLIGHT,
        )
        plt.show()

    # Optional: Measure how long it takes
    if timeit:
        times = []
        for _ in range(100):
            # fmt: off
            start = time()
            kp2_uv_proj = warp_image_batch(kp2_uv, batch.params2, batch.pose2, batch.params1, batch.pose1, (width, height))
            match_keypoints_2d_batch(kp1_uv, kp2_uv_proj, conv_size, distance_threshold, distance=metric)  # type: ignore[arg-type]
            times.append(time() - start)
            # fmt: on
        print("Mean time per batch:", torch.tensor(times).mean())
        print("Mean time (single): ", torch.tensor(times).mean() / batch_size)

    return None


if __name__ == "__main__":
    app()
