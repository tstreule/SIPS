import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from sips.data import CameraPosition

# ==============================================================================
# Colors

_RED, _GREEN, _BLUE, _GRAY = "#e6194B", "#3cb44b", "#4363d8", "#a9a9a9"
COLOR_SONAR_1 = _BLUE
COLOR_SONAR_2 = _GREEN
COLOR_HIGHLIGHT = _RED
COLOR_BACKGROUND = _GRAY
COLOR_SEQUENCE = [_RED, _GREEN, _BLUE, _GRAY]


# ==============================================================================
# Sonar Projection Arcs


def plot_arcs_2d(
    keypoints_uv: npt.ArrayLike,
    image_resolution: tuple[int, int],
    convolution_size: int,
    colors: str | list[str] | None = None,
    keypoint_pairs: tuple[npt.ArrayLike, npt.ArrayLike] | None = None,
    keypoint_pairs_color: str | None = None,
) -> plt.Axes:
    """
    Plot keypoint positions in sonar image (uv) space.

    Parameters
    ----------
    keypoints_uv : npt.ArrayLike
        Keypoints of shape (..., 2).
    image_resolution : tuple[int, int]
        Width and height of image.
    convolution_size : int
        Distance of visual guidelines / xticks and yticks grid.
    colors : str | list[str] | None, optional
        Colors of the keypoints, by default None
    keypoint_pairs : tuple[npt.ArrayLike, npt.ArrayLike] | None, optional
        Keypoint pairs that will be connected by a line, by default None
    keypoint_pairs_color : str | None, optional
        Color of the connecting line for keypoint pairs, by default None

    Returns
    -------
    plt.Axes
        Axes of the plot.

    """
    # Handle input
    keypoints_uv = [
        np.asarray(kp_uv, np.float64).reshape(-1, 2) for kp_uv in keypoints_uv  # type: ignore
    ]
    assert all(kp_uv.shape[-1] == 2 and kp_uv.ndim == 2 for kp_uv in keypoints_uv)
    # Handle colors
    if colors is None:
        colors = COLOR_SEQUENCE
    elif isinstance(colors, str):
        colors = [colors] * len(keypoints_uv)

    width, height = image_resolution

    # Set up figure
    fig = plt.figure()
    ax = fig.add_subplot()
    x_ticks = np.arange(0, width + 1, convolution_size)
    y_ticks = np.arange(0, height + 1, convolution_size)
    ax.set_xticks(x_ticks, minor=True)
    ax.set_yticks(y_ticks, minor=True)
    ax.grid(which="minor", alpha=0.2)
    ax.set_xticks(x_ticks[::4])
    ax.set_yticks(y_ticks[::4])
    ax.grid(which="major", alpha=0.5)
    ax.set_xlim(-convolution_size / 2, width + convolution_size / 2)
    ax.set_ylim(-convolution_size / 2, height + convolution_size / 2)
    ax.set_aspect("equal")

    # Plot points
    ax.invert_yaxis()
    for kp_uv, color in zip(keypoints_uv, colors):
        ax.scatter(kp_uv[:, 0], kp_uv[:, 1], color=color, alpha=0.2)

    # Plot lines
    if keypoint_pairs is not None:
        lines = np.concatenate(keypoint_pairs, axis=-1).reshape(-1, 4)
        for points in lines:
            points = points.reshape(2, 2)
            ax.plot(points[:, 0], points[:, 1], keypoint_pairs_color)

    fig.tight_layout()
    return ax


def plot_arcs_3d(
    keypoints_xyz: npt.ArrayLike,
    camera_pos: CameraPosition | list[CameraPosition | None] | None = None,
    colors: str | list[str] | None = None,
) -> plt.Axes:
    """
    Plot projection arcs in Euclidean (xyz) space.

    Parameters
    ----------
    keypoints_xyz : npt.ArrayLike
        Keypoints of shape (..., n_elevations, 3).
    camera_pos : CameraPosition | list[CameraPosition  |  None] | None, optional
        Camera position for the (set of) keypoints, by default None
    colors : str | list[str] | None, optional
        Colors of the keypoints, by default None

    Returns
    -------
    plt.Axes
        Axes of the plot.

    """
    # Handle input
    keypoints_xyz = [np.asarray(kp_xyz, np.float64) for kp_xyz in keypoints_xyz]  # type: ignore
    assert all(kp_xyz.shape[-1] == 3 and kp_xyz.ndim == 3 for kp_xyz in keypoints_xyz)
    # Camera positions
    camera_positions = np.asarray(camera_pos)
    if camera_positions.ndim == 0:
        camera_positions = camera_positions.repeat(len(keypoints_xyz))
    # Colors
    if colors is None:
        colors = COLOR_SEQUENCE
    elif isinstance(colors, str):
        colors = [colors] * len(keypoints_xyz)

    # Begin figure
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    for color, pos, arcs in zip(colors, camera_positions, keypoints_xyz):
        # Plot camera location
        if isinstance(pos, CameraPosition):
            ax.scatter(pos.x, pos.y, pos.z, marker="o", color=color)
        # Plot arcs
        for arc in arcs:
            kwargs = dict(color=color, alpha=0.2, linewidth=1.8)
            ax.plot(arc[:, 0], arc[:, 1], arc[:, 2], **kwargs)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_aspect("equal")

    fig.tight_layout()

    return ax
