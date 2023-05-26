"""
Tools for visualizing camear poses.

"""
import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from drone_movements._data import CameraPoseTracker, MovableCameraPose

__all__ = ["CameraPoseVisualizer", "plot_camera_poses"]


class CameraPoseVisualizer:
    # Partial credits: https://github.com/demul/extrinsic2pyramid
    """
    3D visualizer for camera poses.

    Parameters
    ----------
    visual_range : float, optional
        Visual range of the sonar, by default 5
    azimuth_fov : float, optional
        Azimuth FOV of the sonar, by default 30
    elevation_fov : float, optional
        Elevation FOV of the sonar, by default 15
    degrees : bool, optional
        If True, will assume 'azimuth_fov' and 'elevation_fov' to be in degrees, by default True
    xlim : tuple[float, float], optional
        Plot limits for x-axis, by default (-20, 20)
    ylim : tuple[float, float], optional
        Plot limits for y-axis, by default (-20, 20)
    zlim : tuple[float, float], optional
        Plot limits for z-axis, by default (-20, 20)
    figsize : tuple[float, float], optional
        Plot size, by default (7, 7)
    """

    def __init__(
        self,
        visual_range: float = 0.3,
        azimuth_fov: float = 30,
        elevation_fov: float = 15,
        degrees: bool = True,
        xlim: tuple[float, float] = (-1, 1),
        ylim: tuple[float, float] = (-1, 1),
        zlim: tuple[float, float] = (-1, 1),
        figsize: tuple[float, float] = (7, 7),
    ):
        # Define length and width of pyramid (i.e., camera view)
        self.visual_range = visual_range
        if degrees:
            azimuth_fov *= np.pi / 180
            elevation_fov *= np.pi / 180
        self.azimuth_fov = azimuth_fov
        self.elevation_fov = elevation_fov
        # Define figure and axes
        self.fig = plt.figure(figsize=figsize)
        self.ax = plt.axes(projection="3d")
        self.ax.set_aspect("auto")
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)  # type: ignore
        self.ax.set_zlim(zlim)  # type: ignore
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_zlabel("z")  # type: ignore

    def plot(
        self,
        pose: MovableCameraPose | CameraPoseTracker,
        color: str | list[str] = "auto",
        alpha: float | list[float] = 0.35,
        title: str | None = None,
    ) -> plt.Axes:
        """
        Plot and show the camera pose(s).

        Parameters
        ----------
        pose : MovableCameraPose | CameraPoseTracker
            Camera pose(s) to be plotted
        color : str | list[str], optional
            Color of the camera, by default "auto"
            If "auto", uses a colormap to indicate movements in time.
            If string, every camera pose has the given color.
            If list of strings, defines the color of all poses. Length must be same as
            number of different poses.
        alpha : float | list[float], optional
            Transparency of the plotted camera pose(s), by default 0.35
        title : str | None, optional
            Plot title, by default None

        Returns
        -------
        plt.Axes

        """
        if not isinstance(pose, CameraPoseTracker):
            pose = CameraPoseTracker(pose)
        if not isinstance(alpha, list):
            alpha = [alpha for _ in range(len(pose))]

        if color == "auto":  # auto coloring
            colormap = cm.gist_rainbow  # type: ignore  # "matplotlib.colors.Colormap"
            step_size = max(1, min(colormap.N // len(pose), 15))
            color_arr = colormap(range(0, len(pose) * step_size, step_size))
            color = color_arr[:, :3].tolist()  # don't use alpha from colormap
        if not isinstance(color, list):
            color = [color for _ in range(len(pose))]

        for (p, _), c, a in zip(pose, color, alpha):
            self._plot(p, c, a)

        self.ax.set_title(title or "Camera Poses")
        self.ax.set_aspect("equal")
        return self.ax

    def _plot(
        self, pose: MovableCameraPose, color: str | list[float], alpha: float
    ) -> None:
        # Define pyramid vertices
        azimuth = self.visual_range * np.tan(self.azimuth_fov / 2)
        elevation = self.visual_range * np.tan(self.elevation_fov / 2)
        vertex_std = np.array(  # camera is looking along the x-axis
            [
                [0, 0, 0, 1],
                [self.visual_range, azimuth, elevation, 1],
                [self.visual_range, azimuth, -elevation, 1],
                [self.visual_range, -azimuth, -elevation, 1],
                [self.visual_range, -azimuth, elevation, 1],
            ]
        )
        # Adjust the pose of the vertices
        vertex_trsf = (vertex_std @ pose.as_extrinsic().T)[:, :-1]
        meshes = [
            [vertex_trsf[0], vertex_trsf[1], vertex_trsf[2]],
            [vertex_trsf[0], vertex_trsf[2], vertex_trsf[3]],
            [vertex_trsf[0], vertex_trsf[3], vertex_trsf[4]],
            [vertex_trsf[0], vertex_trsf[4], vertex_trsf[1]],
            # [vertex_trsf[1], vertex_trsf[2], vertex_trsf[3], vertex_trsf[4]],
        ]
        self.ax.add_collection3d(  # type: ignore
            Poly3DCollection(
                meshes, facecolors=color, linewidths=0.3, edgecolors=color, alpha=alpha
            )
        )


def plot_camera_poses(
    pose_or_tracker: MovableCameraPose | CameraPoseTracker,
    auto_settings: bool = True,
    show: bool = True,
    **kwargs,
) -> plt.Axes:

    # Assure tracker instance
    tracker = (
        pose_or_tracker
        if isinstance(pose_or_tracker, CameraPoseTracker)
        else CameraPoseTracker(pose_or_tracker)
    )

    if auto_settings:
        # Define plotter settings
        xlim, ylim, zlim = tracker.get_pos_limits()
        xlen = xlim[1] - xlim[0]
        ylen = ylim[1] - ylim[0]
        zlen = zlim[1] - zlim[0]
        minlen = 0.5  # assert minimal length
        maxlen = max(xlen, ylen, zlen, minlen)
        xlim = (xlim[0] - (maxlen - xlen) / 2, xlim[1] + (maxlen - xlen) / 2)
        ylim = (ylim[0] - (maxlen - ylen) / 2, ylim[1] + (maxlen - ylen) / 2)
        zlim = (zlim[0] - (maxlen - zlen) / 2, zlim[1] + (maxlen - zlen) / 2)

        # Instantiate visualizer
        vis = CameraPoseVisualizer(
            0.1 * maxlen, xlim=xlim, ylim=ylim, zlim=zlim, **kwargs
        )

    else:
        # Instantiate visualizer and plot
        vis = CameraPoseVisualizer(**kwargs)

    # Plot
    ax = vis.plot(tracker)
    if show:
        plt.show()

    return ax
