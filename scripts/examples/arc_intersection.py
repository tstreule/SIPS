import torch
from matplotlib import pyplot as plt

from scripts.examples._sonar_data import get_random_datum_tuple
from sips.utils.keypoint_matching import match_keypoints_2d
from sips.utils.plotting import (
    COLOR_HIGHLIGHT,
    COLOR_SONAR_1,
    COLOR_SONAR_2,
    plot_arcs_2d,
    plot_arcs_3d,
)
from sips.utils.point_projection import uv_to_xyz, xyz_to_uv


def main():
    # Set data
    sonar_tuple = get_random_datum_tuple(seed=123)
    sonar1 = sonar_tuple.sonar1
    sonar2 = sonar_tuple.sonar2
    width, height = sonar1.image.shape

    # Create convoolutional grid
    conv_size = 8
    xs = torch.arange(0, width, conv_size) + conv_size / 2
    ys = torch.arange(0, height, conv_size) + conv_size / 2
    mesh = torch.meshgrid(xs, ys)
    grid = torch.stack(mesh, dim=-1).reshape(-1, 2).to(torch.float32)
    # Define (random) keypoints for both images
    alpha = 0.7
    torch.random.manual_seed(420)
    std = torch.broadcast_to(torch.tensor(alpha * conv_size), grid.shape)
    kp1_uv = grid + torch.normal(0, std)
    kp2_uv = grid + torch.normal(0, std)

    # Arc position
    kp1_xyz = uv_to_xyz(kp1_uv, sonar1)
    kp2_xyz = uv_to_xyz(kp2_uv, sonar2)

    # Point projection to other camera
    kp2_uv_proj = xyz_to_uv(kp2_xyz.reshape(-1, 3), sonar1)
    kp2_uv_proj = kp2_uv_proj.reshape(*kp2_xyz.shape[:2], 2)

    # Find matches that are close
    distance_threshold = 5
    matches_uv = match_keypoints_2d(
        kp1_uv.reshape(*mesh[0].shape, 2).to("mps"),
        kp2_uv_proj.to("mps"),
        conv_size,
        distance_threshold=distance_threshold,
    )
    assert (
        torch.linalg.vector_norm(matches_uv[0] - matches_uv[1], dim=1).max()
        < distance_threshold
    )

    # Plot point projection
    kp2_xyz_filtered = kp2_xyz.clone()
    nan_mask = torch.isnan(kp2_uv_proj).any(-1)
    kp2_xyz_filtered[nan_mask] = torch.nan
    plot_arcs_3d(
        [kp1_xyz[::12], kp2_xyz[::12], kp2_xyz_filtered[::12]],
        [sonar1.pose.position, sonar2.pose.position, None],
        colors=[COLOR_SONAR_1, COLOR_SONAR_2, COLOR_HIGHLIGHT],
    )
    plot_arcs_2d(
        [kp1_uv, kp2_uv_proj],
        (width, height),
        conv_size,
        [COLOR_SONAR_1, COLOR_SONAR_2],
        (matches_uv[0].cpu(), matches_uv[1].cpu()),
        COLOR_HIGHLIGHT,
    )
    plt.show()

    return None


if __name__ == "__main__":
    main()
