"""
Script to plan and store FLS drone movements in a CSV file.

Goal
----
Plan movements in order to obtain sonar camera data that enables keypoint matching 
between subsequent image frames.

Notes
-----
Because of the sonar construction, it is not possible to disambiguate the angle of the 
reflected echo along the elevation arc (Hurtós Vilarnau, 2014).  In optical imaging it 
is common to use homographies to approximate viewpoint changes.  Similar transformations 
can be used in acoustic imaging (Negahdaripour, 2012).  However, due to the nature of 
FLS (e.g., the usually small elevation angle) an object can be quickly out of view when 
the camera is moved.  We want to limit the camera movements to planar rotations and 
translations, in order to obtain a proper dataset where two images from subsequent 
frames include a high percentage of the same keypoints.

"""
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
from matplotlib import cm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation as R
from typing_extensions import Self


class CameraPose:
    """
    Incorporates an extrinsic parameter matrix for camera pose.

    Parameters
    ----------
    extrinsic_matrix : npt.NDArray[np.float_] | None, optional
        Extrinsic 4x4 camera matrix, by default None
    """

    def __init__(self, extrinsic_matrix: npt.NDArray[np.float_] | None = None):
        self.extrinsic_matrix = np.eye(4)
        if extrinsic_matrix is not None:
            assert extrinsic_matrix.shape == (4, 4)
            self.extrinsic_matrix[:] = extrinsic_matrix

    @staticmethod
    def from_vector(
        vector: npt.NDArray[np.float_], degrees: bool = True
    ) -> "CameraPose":
        assert vector.shape == (6,)
        pose = CameraPose()
        pose.position = vector[:3]
        pose.rotation = R.from_rotvec(vector[3:], degrees=degrees).as_matrix()  # type: ignore
        return pose

    # ----------------------------------------------------------------------
    # Properties

    @property
    def position(self) -> npt.NDArray[np.float_]:
        return self.extrinsic_matrix[:-1, -1]

    @position.setter
    def position(self, position: npt.NDArray[np.float_]):
        self.extrinsic_matrix[:-1, -1] = position

    @property
    def rotation(self) -> npt.NDArray[np.float_]:
        return self.extrinsic_matrix[:3, :3]

    @rotation.setter
    def rotation(self, rotation: npt.NDArray[np.float_]):
        rot = R.from_matrix(rotation).as_matrix()  # gives a normalized matrix
        if not np.allclose(rotation, rot) or rotation.shape != (3, 3):
            raise ValueError("Invalid rotation matrix")
        self.extrinsic_matrix[:3, :3] = rotation

    # ----------------------------------------------------------------------
    # Utils

    def copy(self) -> "CameraPose":
        return CameraPose(self.as_matrix().copy())

    def as_matrix(self) -> npt.NDArray[np.float_]:
        return self.extrinsic_matrix

    def as_vector(self, degrees: bool = True) -> npt.NDArray[np.float_]:
        return np.concatenate([self.position, self.get_rotvec(degrees=degrees)])

    def get_rotvec(self, degrees: bool = True) -> npt.NDArray[np.float_]:
        return R.from_matrix(self.rotation).as_rotvec(degrees=degrees)  # type: ignore

    def set_rotvec(self, rotvec: npt.ArrayLike, degrees: bool = True) -> Self:
        self.rotation = R.from_rotvec(rotvec, degrees=degrees).as_matrix()  # type: ignore
        return self

    def get_quat(self) -> npt.NDArray[np.float_]:
        return R.from_matrix(self.rotation).as_quat()

    # ----------------------------------------------------------------------
    # Absolute transformations

    def translate(self, x: float = 0, y: float = 0, z: float = 0) -> Self:
        self.position = self.position + np.array([x, y, z])
        return self

    def rotate(
        self, x: float = 0, y: float = 0, z: float = 0, degrees: bool = True
    ) -> Self:
        rot = R.from_euler("xyz", [x, y, z], degrees=degrees)
        self.rotation = rot.apply(self.rotation, inverse=True)
        return self

    # ----------------------------------------------------------------------
    # Relative transformations

    def surge(self, dist: float) -> Self:
        self.position += self.rotation @ np.array([dist, 0, 0])
        return self

    def sway(self, dist: float) -> Self:
        self.position += self.rotation @ np.array([0, dist, 0])
        return self

    def heave(self, dist: float) -> Self:
        self.position += self.rotation @ np.array([0, 0, dist])
        return self

    def roll(self, angle: float, degrees: bool = True) -> Self:
        return self.rotate(x=angle, degrees=degrees)

    def pitch(self, angle: float, degrees: bool = True) -> Self:
        return self.rotate(y=angle, degrees=degrees)

    def yaw(self, angle: float, degrees: bool = True) -> Self:
        return self.rotate(z=angle, degrees=degrees)


@dataclass
class RelativeMovement:
    surge: float
    sway: float
    heave: float
    roll: float
    pitch: float
    yaw: float
    degrees: bool = True  # with default value

    def __post_init__(self):
        if not self.degrees:
            self._rad2deg()

    def _rad2deg(self) -> Self:
        assert not self.degrees
        self.roll *= 180 / np.pi
        self.pitch *= 180 / np.pi
        self.yaw *= 180 / np.pi
        self.degrees = True
        return self

    def _deg2rad(self) -> Self:
        assert self.degrees
        self.roll *= np.pi / 180
        self.pitch *= np.pi / 180
        self.yaw *= np.pi / 180
        self.degrees = False
        return self

    def to_list(self, degrees: bool = True) -> list[float]:
        if degrees is self.degrees:  # internal and requested coincide
            return [self.surge, self.sway, self.heave, self.roll, self.pitch, self.yaw]

        if degrees and not self.degrees:
            out = self._rad2deg().to_list(degrees=degrees)
            self._deg2rad()
        else:
            out = self._deg2rad().to_list(degrees=degrees)
            self._rad2deg()
        return out


class CameraPoseTracker:
    """
    Keeps track of the movement history of camera poses.

    Parameters
    ----------
    init_pose : CameraPose
        Initial pose
    """

    _abscol = ["x[m]", "y[m]", "z[m]", "x[deg]", "y[deg]", "z[deg]"]
    _relcol = ["surge[m]", "sway[m]", "heave[m]", "roll[deg]", "pitch[deg]", "yaw[deg]"]
    _abscol_rad = [c.replace("[deg]", "[rad]") for c in _abscol]
    _relcol_rad = [c.replace("[deg]", "[rad]") for c in _relcol]

    def __init__(self, init_pose: CameraPose) -> None:
        self.abs_history: list[CameraPose] = [init_pose]
        self.rel_history: list[RelativeMovement] = [RelativeMovement(0, 0, 0, 0, 0, 0)]

    def __len__(self) -> int:
        assert len(self.abs_history) == len(self.rel_history)
        return len(self.abs_history)

    def __getitem__(self, ix: int) -> tuple[CameraPose, RelativeMovement]:
        return self.abs_history[ix], self.rel_history[ix]

    def __iter__(self):
        self._current_index = 0
        return self

    def __next__(self):
        ix = self._current_index
        if ix > len(self) - 1:
            raise StopIteration
        self._current_index += 1
        return self[ix]

    # ----------------------------------------------------------------------
    # Transformations

    def move(
        self,
        surge: float = 0,
        sway: float = 0,
        heave: float = 0,
        roll: float = 0,
        pitch: float = 0,
        yaw: float = 0,
        degrees: bool = True,
    ) -> Self:
        """
        Movements relative to the camera's orientation.

        """
        curr_pose = self.abs_history[-1]
        next_pose = curr_pose.copy()
        # Apply transformation
        next_pose.surge(surge).sway(sway).heave(heave)
        next_pose.roll(roll, degrees).pitch(pitch, degrees).yaw(yaw, degrees)
        # Append to history
        self.abs_history.append(next_pose)
        self.rel_history.append(
            RelativeMovement(surge, sway, heave, roll, pitch, yaw, degrees)
        )
        return self

    # ----------------------------------------------------------------------
    # Export and import

    def as_pandas(
        self, select: str | None = None, degrees: bool = True
    ) -> pd.DataFrame:
        # Read out absolute movements
        absolute = pd.DataFrame(
            [p.as_vector(degrees=degrees) for p in self.abs_history],
            columns=self._abscol if degrees else self._abscol_rad,
        )
        # Read out relative movements
        relative = pd.DataFrame(
            [m.to_list(degrees=degrees) for m in self.rel_history],
            columns=self._relcol if degrees else self._relcol_rad,
        )

        if select == "absolute":
            return absolute
        elif select == "relative":
            return relative
        else:
            return pd.concat([absolute, relative], axis=1)

    @staticmethod
    def from_pandas(data: pd.DataFrame, degrees: bool = True) -> "CameraPoseTracker":
        if degrees:
            abs_cols = CameraPoseTracker._abscol
            rel_cols = CameraPoseTracker._relcol
        else:
            abs_cols = CameraPoseTracker._abscol_rad
            rel_cols = CameraPoseTracker._relcol_rad
        # Pop first row
        init_row = data.iloc[0]
        data = data.iloc[1:]
        # Initialize tracker
        init_pose_np = init_row[abs_cols].to_numpy()
        init_pose = CameraPose.from_vector(init_pose_np, degrees)
        poses = CameraPoseTracker(init_pose)
        # Extract all other rows
        for _, row in data.iterrows():
            absolute = row[abs_cols].to_numpy()
            relative = row[rel_cols].to_list() + [degrees]
            poses.abs_history.append(CameraPose.from_vector(absolute, degrees))
            poses.rel_history.append(RelativeMovement(*relative))
        return poses

    def to_csv(
        self,
        filepath: str | Path,
        sep: str = ",",
        select: str | None = None,
        degrees: bool = True,
    ) -> None:
        self.as_pandas(select, degrees).to_csv(filepath, index=False, sep=sep)


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
        self.ax = plt.axes(projection="3d", aspect="equal")
        self.ax.set_aspect("auto")
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)  # type: ignore
        self.ax.set_zlim(zlim)  # type: ignore
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_zlabel("z")  # type: ignore

    def plot(
        self,
        pose: CameraPose | CameraPoseTracker,
        color: str | list[str] = "auto",
        alpha: float | list[float] = 0.35,
        title: str | None = None,
    ) -> Self:
        """
        Plot and show the camera pose(s).

        Parameters
        ----------
        pose : CameraPose | CameraPoseTracker
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
        CameraPoseVisualizer

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

        plt.title(title or "Camera Poses")
        plt.show()
        return self

    def _plot(self, pose: CameraPose, color: str | list[float], alpha: float) -> Self:
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
        vertex_trsf = (vertex_std @ pose.as_matrix().T)[:, :-1]
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
        return self


# ==============================================================================
# Scheduling


def make_dummy_poses() -> CameraPoseTracker:
    """
    Make a fixed sequence of poses for testing purposes.

    """
    poses = CameraPoseTracker(init_pose=CameraPose())
    # poses = CameraPoseTracker(init_pose=CameraPose().roll(45))

    for i in range(24):
        # Makes two circles, one counter-clockwise and one clockwise
        if i < 12:
            poses.move(surge=0.4, yaw=30)
        else:
            poses.move(surge=0.4, yaw=-30)

    return poses


class PosePlanner3D:
    """
    Planner for random camera poses.

    Parameters
    ----------
    x_lim : float | tuple[float, float], optional
        Limits left and right position, by default (-1, 1)
    y_lim : float | tuple[float, float], optional
        Limits front and back position, by default (-1, 1)
    z_lim : float | tuple[float, float], optional
        Limits top and bottom position, by default (-1, 1)
    rot_x_lim : float | tuple[float, float], optional
        Limits rotation around x-axis, by default (-45, 45)
    rot_y_lim : float | tuple[float, float], optional
        Limits rotation anround y-axis, by default (-45, 45)
    rot_z_lim : float | tuple[float, float], optional
        Limits rotation around z-axis, by default (-45, 45)
    max_surge : float, optional
        Maximal surge between two subsequent camera poses, by default 0.5
    max_sway : float, optional
        Maximal sway between two subsequent camera poses, by default 0.5
    max_heave : float, optional
        Maximal heave between two subsequent camera poses, by default 0.5
    max_roll : float, optional
        Maximal roll between two subsequent camera poses, by default 10
    max_pitch : float, optional
        Maximal pitch between two subsequent camera poses, by default 10
    max_yaw : float, optional
        Maximal yaw between two subsequent camera poses, by default 10
    degrees : bool, optional
        If True, it assumes that 'yawlim' and 'max_yaw' are in degrees, by default True
    seed : int | None, optional
        Set the initial random state for reproducability, by default None
    verbose : int, optional
        For debugging, by default 0
    """

    def __init__(
        self,
        x_lim: float | tuple[float, float] = (-1, 1),
        y_lim: float | tuple[float, float] = (-1, 1),
        z_lim: float | tuple[float, float] = (-1, 1),
        rot_x_lim: float | tuple[float, float] = (-45, 45),
        rot_y_lim: float | tuple[float, float] = (-45, 45),
        rot_z_lim: float | tuple[float, float] = (-45, 45),
        max_surge: float = 0.5,
        max_sway: float = 0.5,
        max_heave: float = 0.5,
        max_roll: float = 10,
        max_pitch: float = 10,
        max_yaw: float = 10,
        degrees: bool = True,
        seed: int | None = None,
        verbose: int = 0,
    ):
        # Set camera position and angle boundaries
        self.x_lim = self._check_lim(x_lim)
        self.y_lim = self._check_lim(y_lim)
        self.z_lim = self._check_lim(z_lim)
        self.rot_x_lim = self._check_rot_lim(rot_x_lim, degrees)
        self.rot_y_lim = self._check_rot_lim(rot_y_lim, degrees)
        self.rot_z_lim = self._check_rot_lim(rot_z_lim, degrees)

        # Set movability parameters
        self.max_surge = abs(max_surge)
        self.max_sway = abs(max_sway)
        self.max_heave = abs(max_heave)
        if not degrees:
            max_roll *= 180 / np.pi
            max_pitch *= 180 / np.pi
            max_yaw *= 180 / np.pi
        self.max_roll = abs(max_roll)
        self.max_pitch = abs(max_pitch)
        self.max_yaw = abs(max_yaw)

        # Set initial state
        self.cur_pose = CameraPose()

        # Other settings
        # self.degrees = True
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.verbose = verbose

    def _check_lim(self, lim: float | tuple[float, float]) -> tuple[float, float]:
        if isinstance(lim, (int, float)):
            lower = -abs(lim)
            upper = abs(lim)
        else:
            lower, upper = lim
            assert upper - lower >= 0, "upper - lower < 0"
        return lower, upper

    def _check_rot_lim(
        self, lim: float | tuple[float, float], degrees: bool
    ) -> tuple[float, float]:
        lower, upper = self._check_lim(lim)
        # Convert to degrees
        if not degrees:
            lower *= 180 / np.pi
            upper *= 180 / np.pi
        # Do not exceed [-180, 180] (required for _get_next_{roll, pitch, yaw})
        lower = max(lower, -180)
        upper = min(upper, 180)
        return lower, upper

    def debug(self, *args, **kwargs) -> None:
        """Debug print"""
        if self.verbose > 0:
            print(*args, **kwargs)

    def get_random_poses(
        self, init_pose: CameraPose, n_movements: int = 100
    ) -> CameraPoseTracker:
        """
        Get a random sequence of poses within the allowed boundaries.

        Parameters
        ----------
        init_pose : CameraPose
            Initial camera position.
        n_movements : int, optional
            The amount of poses to generate, by default 100

        Returns
        -------
        CameraPoseTracker
            Random sequence of poses within the allowed boundaries.

        """
        poses = CameraPoseTracker(init_pose)

        for i in range(n_movements - 1):  # the first pose is already given
            self.debug("i:", i)
            # Get random movement within the allowed limits.
            # Note that we have to follow the same update order as in move_rel().
            # Otherwise, we will have wrong estimates for the current relative pose.
            move_kwargs = self._get_next_movement(poses)
            poses.move(**move_kwargs, degrees=True)

        return poses

    def _get_next_movement(self, poses: CameraPoseTracker) -> dict[str, float]:
        # Debug print to check whether current position estimate is correct.
        # Works only if the initial pose starts at (x, y, z) = (0, 0, 0) and has
        # no initial rotations.
        self.debug("estimate:", self.cur_pose.as_vector(degrees=True).round(4))
        self.debug("true:    ", poses.abs_history[-1].as_vector(degrees=True).round(4))

        # Get and update surge
        surge = self._get_next_surge()
        self.cur_pose.surge(surge)
        self.debug(f"    -> surge={surge:>9.4f}:", self.cur_pose.position.round(4))

        # Get and update sway
        sway = self._get_next_sway()
        self.cur_pose.sway(sway)
        self.debug(f"    -> sway ={sway:>9.4f}:", self.cur_pose.position.round(4))

        # Get and update heave
        heave = self._get_next_heave()
        self.cur_pose.heave(heave)
        self.debug(f"    -> heave={heave:>9.4f}:", self.cur_pose.position.round(4))

        # Get and update roll
        roll = self._get_next_roll()
        self.cur_pose.roll(roll, degrees=True)
        self.debug(f"    -> roll ={roll:>9.4f}:", self.cur_pose.get_rotvec().round(4))

        # Get and update pitch
        pitch = self._get_next_pitch()
        self.cur_pose.pitch(pitch, degrees=True)
        self.debug(f"    -> pitch={pitch:>9.4f}:", self.cur_pose.get_rotvec().round(4))

        # Get and update yaw
        yaw = self._get_next_yaw()
        self.cur_pose.yaw(yaw, degrees=True)
        self.debug(f"    -> yaw  ={yaw:>9.4f}:", self.cur_pose.get_rotvec().round(4))

        # Check
        cur_rotvec = self.cur_pose.get_rotvec(degrees=True)
        assert self.rot_x_lim[0] <= cur_rotvec[0] <= self.rot_x_lim[1]
        assert self.rot_y_lim[0] <= cur_rotvec[1] <= self.rot_y_lim[1]
        assert self.rot_z_lim[0] <= cur_rotvec[2] <= self.rot_z_lim[1]

        next_move = {
            "surge": surge,
            "sway": sway,
            "heave": heave,
            "roll": roll,
            "pitch": pitch,
            "yaw": yaw,
        }
        return next_move

    def _get_min_max_translation(self) -> tuple[float, float]:
        # Raise error if outside of allowed boundary box
        pos_x, pos_y, pos_z = self.cur_pose.position
        if (
            not self.x_lim[0] <= pos_x <= self.x_lim[1]
            or not self.y_lim[0] <= pos_y <= self.y_lim[1]
            or not self.z_lim[0] <= pos_z <= self.z_lim[1]
        ):
            raise RuntimeError("Camera is outside of the allowed boundaries")

        # Get direction of movement (assuming we point towards the x-axis)
        direction = self.cur_pose.rotation[:, 0]

        # Don't collide with front/back border
        x_plane_normal = np.array([1, 0, 0])
        x0_plane_point = np.array([self.x_lim[0], 0, 0])
        x1_plane_point = np.array([self.x_lim[1], 0, 0])
        min_x = -self._get_directed_dist_to_plane(x_plane_normal, x0_plane_point)
        max_x = self._get_directed_dist_to_plane(x_plane_normal, x1_plane_point)
        if direction[0] < 0:
            min_x, max_x = -max_x, -min_x

        # Don't collide with left/right borders
        y_plane_normal = np.array([0, 1, 0])
        y0_plane_point = np.array([0, self.y_lim[0], 0])
        y1_plane_point = np.array([0, self.y_lim[1], 0])
        min_y = -self._get_directed_dist_to_plane(y_plane_normal, y0_plane_point)
        max_y = self._get_directed_dist_to_plane(y_plane_normal, y1_plane_point)
        if direction[1] < 0:
            min_y, max_y = -max_y, -min_y

        # Don't collide with top/bottom border
        z_plane_normal = np.array([0, 0, 1])
        z0_plane_point = np.array([0, 0, self.z_lim[0]])
        z1_plane_point = np.array([0, 0, self.z_lim[1]])
        min_z = -self._get_directed_dist_to_plane(z_plane_normal, z0_plane_point)
        max_z = self._get_directed_dist_to_plane(z_plane_normal, z1_plane_point)
        if direction[2] < 0:
            min_z, max_z = -max_z, -min_z

        # Calculate min/max translation
        min_ = max(min_x, min_y, min_z)
        max_ = min(max_x, max_y, max_z)
        return min_, max_

    def _get_directed_dist_to_plane(
        self,
        plane_normal: npt.NDArray[np.float_],
        plane_point: npt.NDArray[np.float_],
        epsilon: float = 1e-6,
    ):
        # Get current camera view direction
        camera_point = self.cur_pose.position
        camera_direction = self.cur_pose.rotation @ np.array([1, 0, 0])

        # Get intersecting point of ray and plane
        denom = plane_normal.dot(camera_direction)
        if abs(denom) < epsilon:
            return np.inf
        si = -plane_normal.dot(camera_point - plane_point) / denom
        psi = camera_point + si * camera_direction

        # Calculate distance
        return np.sqrt(np.sum((camera_point - psi) ** 2))

    def _get_next_surge(self) -> float:
        min_, max_ = self._get_min_max_translation()
        min_ = max(min_, -self.max_surge)
        max_ = min(max_, self.max_surge)
        return self.rng.uniform(min_, max_)

    def _get_next_sway(self) -> float:
        self.cur_pose.yaw(90, degrees=True)
        min_, max_ = self._get_min_max_translation()
        self.cur_pose.yaw(-90, degrees=True)
        min_ = max(min_, -self.max_sway)
        max_ = min(max_, self.max_sway)
        return self.rng.uniform(min_, max_)

    def _get_next_heave(self) -> float:
        self.cur_pose.pitch(-90, degrees=True)
        min_, max_ = self._get_min_max_translation()
        self.cur_pose.pitch(90, degrees=True)
        min_ = max(min_, -self.max_heave)
        max_ = min(max_, self.max_heave)
        return self.rng.uniform(min_, max_)

    def _get_next_roll(self) -> float:
        # Estimate effect of roll on rotation vector
        cur_rotvec = self.cur_pose.get_rotvec(degrees=True)
        oth_rotvec = self.cur_pose.copy().roll(1).get_rotvec(degrees=True)
        delta = oth_rotvec - cur_rotvec

        # Find min/max roll to stay within allowed rotation parameters
        with np.errstate(divide="ignore"):
            rot_min_max = (
                np.array([self.rot_x_lim, self.rot_y_lim, self.rot_z_lim])
                - cur_rotvec[:, None]
            ) / (delta[:, None])
        mask = np.array([[True, False] if val >= 0 else [False, True] for val in delta])
        min_ = max(0.9 * np.nanmax(rot_min_max[mask]), -self.max_roll)  # type: ignore
        max_ = min(0.9 * np.nanmin(rot_min_max[~mask]), self.max_roll)  # type: ignore

        if max_ - min_ <= 0:
            return 0
        return self.rng.uniform(min_, max_)

    def _get_next_pitch(self) -> float:
        # Estimate effect of pitch on rotation vector
        cur_rotvec = self.cur_pose.get_rotvec(degrees=True)
        oth_rotvec = self.cur_pose.copy().pitch(1).get_rotvec(degrees=True)
        delta = oth_rotvec - cur_rotvec

        # Find min/max pitch to stay within allowed rotation parameters
        with np.errstate(divide="ignore"):
            rot_min_max = (
                np.array([self.rot_x_lim, self.rot_y_lim, self.rot_z_lim])
                - cur_rotvec[:, None]
            ) / (delta[:, None])
        mask = np.array([[True, False] if val >= 0 else [False, True] for val in delta])
        min_ = max(0.9 * np.nanmax(rot_min_max[mask]), -self.max_pitch)  # type: ignore
        max_ = min(0.9 * np.nanmin(rot_min_max[~mask]), self.max_pitch)  # type: ignore

        if max_ - min_ <= 0:
            return 0
        return self.rng.uniform(min_, max_)

    def _get_next_yaw(self) -> float:
        # Estimate effect of yaw on rotation vector
        cur_rotvec = self.cur_pose.get_rotvec(degrees=True)
        oth_rotvec = self.cur_pose.copy().yaw(1, degrees=True).get_rotvec(degrees=True)
        delta = oth_rotvec - cur_rotvec

        # Find min/max yaw to stay within allowed rotation parameters
        with np.errstate(divide="ignore"):
            rot_min_max = (
                np.array([self.rot_x_lim, self.rot_y_lim, self.rot_z_lim])
                - cur_rotvec[:, None]
            ) / (delta[:, None])
        mask = np.array([[True, False] if val >= 0 else [False, True] for val in delta])
        min_ = max(0.9 * np.nanmax(rot_min_max[mask]), -self.max_yaw)  # type: ignore
        max_ = min(0.9 * np.nanmin(rot_min_max[~mask]), self.max_yaw)  # type: ignore

        if max_ - min_ <= 0:
            return 0
        return self.rng.uniform(min_, max_)


class PosePlanner2D(PosePlanner3D):
    """
    Planar planner for random camera poses.
    """

    def __init__(
        self,
        x_lim: float | tuple[float, float] = (-1, 1),
        y_lim: float | tuple[float, float] = (-1, 1),
        rot_lim: float | tuple[float, float] = (-45, 45),
        max_surge: float = 0.5,
        max_sway: float = 0.5,
        max_yaw: float = 10,
        degrees: bool = True,
        seed: int | None = None,
        verbose: int = 0,
    ):
        super().__init__(
            # fmt: off
            x_lim=x_lim, y_lim=y_lim, z_lim=0,
            rot_x_lim=0, rot_y_lim=0, rot_z_lim=rot_lim,
            max_surge=max_surge, max_sway=max_sway, max_heave=0,
            max_pitch=0, max_roll=0, max_yaw=max_yaw,
            degrees=degrees, seed=seed, verbose=verbose
            # fmt: on
        )


# ==============================================================================
# Utils


def convert_camera_to_drone_relative(
    camera_poses: CameraPoseTracker, angle: float = 0, degrees: bool = True
) -> CameraPoseTracker:
    """
    Adjusts the poses to make up for the angle between drone and camera.

    Parameters
    ----------
    camera_poses : CameraPoseTracker
        Camera poses to be converted.
    angle : float, optional
        Adjusting pitch angle between camera and drone orientation, by default 0
    degrees : bool, optional
        Whether 'angle' is given in degrees or radians, by default True

    Returns
    -------
    CameraPoseTracker
        Converted poses.

    """
    if angle == 0:
        return camera_poses

    # Set up new pose tracker while adjusting the initial orientation
    init_drone = camera_poses.abs_history[0].copy().pitch(-angle, degrees=degrees)
    drone_poses = CameraPoseTracker(init_pose=init_drone)

    # Translate the relative camera movements to relative drone movements
    # Note: We skip i=0 since initial pose is already set up
    for i in range(1, len(camera_poses)):

        # Get rotation between camera and drone orientation
        prev_camera_orient = R.from_quat(camera_poses.abs_history[i - 1].get_quat())
        prev_drone_orient = R.from_quat(drone_poses.abs_history[i - 1].get_quat())

        # Find relative drone movement
        delta_camera = camera_poses.rel_history[i]
        camera_mov = delta_camera.to_list()[:3]
        drone_mov = (prev_drone_orient.inv() * prev_camera_orient).apply(camera_mov)

        # Find relative drone rotation
        # Note: Big 'XYZ' correspond to intrinsic rotations in given sequential order
        camera_rot = R.from_euler(
            "XYZ", delta_camera.to_list()[3:], degrees=delta_camera.degrees
        )
        drone_rot = (prev_drone_orient.inv() * camera_rot).as_euler("XYZ")

        # Apply
        surge, sway, heave = drone_mov
        roll, pitch, yaw = drone_rot
        drone_poses.move(surge, sway, heave, roll, pitch, yaw, degrees=False)

        # Assert equal camera and drone position
        np.testing.assert_allclose(
            camera_poses.abs_history[i].position,
            drone_poses.abs_history[i].position,
        )

    return drone_poses


# ==============================================================================
# Main


def main_dummy(plot: bool = False, save_dir: str | Path | None = None) -> None:
    """
    Generate dummy poses, save to file and visualize.

    Parameters
    ----------
    plot : bool, optional
        If True, shows a plot, by default False
    save_dir : str | Path | None, optional
        If str or Path, will store as CSV, by default None

    """
    poses = make_dummy_poses()
    if plot:
        CameraPoseVisualizer(0.35).plot(poses)
    if save_dir:
        Path(save_dir).mkdir(exist_ok=True, parents=True)
        poses.to_csv(Path(save_dir) / "drone_move_dummy.csv")


def main(
    drone_relative: bool = True,
    plot: bool = False,
    save_dir: str | Path | None = None,
    n_movements: int = 100,
    seed=1234,
    verbose: int = 0,
) -> None:
    """
    Generates random poses.

    Parameters
    ----------
    plot : bool, optional
        If True, shows the plot, by default False
    save_dir : str | Path | None, optional
        If str or Path, will save as CSV, by default None
    n_movements : int, optional
        Number of camera movements to generate, by default 100
    seed : int, optional
        Seed for reproducability, by default 1234
    verbose : int, optional
        Verbose printing, by default 0

    """
    # Init pose
    x, y, z = 0, 0, 0
    roll, pitch, yaw = 0, 45, 0
    init_pose = CameraPose().translate(x=x, y=y, z=z).set_rotvec([roll, pitch, yaw])

    # Make random poses
    # pose_planner = PosePlanner3D(
    #     # fmt: off
    #     x_lim=1, y_lim=1, z_lim=0.01,
    #     max_surge=.3, max_sway=.2, max_heave=.1,
    #     rot_x_lim=5, rot_y_lim=5, rot_z_lim=45,
    #     max_roll=1, max_pitch=5, max_yaw=30,
    #     seed=seed, verbose=verbose,
    #     # fmt: on
    # )
    pose_planner = PosePlanner2D(
        # fmt: off
        x_lim=(-1, 0), y_lim=1, rot_lim=15,
        max_surge=0.3, max_sway=0.2, max_yaw=30,
        seed=seed, verbose=verbose,
        # fmt: on
    )
    poses = pose_planner.get_random_poses(init_pose, n_movements=n_movements)

    # Convert movements from camera to drone perspective
    if drone_relative:
        poses = convert_camera_to_drone_relative(poses, angle=45)

    # Save as CSV
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        filename = f"drone_move_{time.time() * 1e3:.0f}.csv"
        poses.to_csv(save_dir / filename, sep="\t")

    # Visualize
    if plot:
        xlim = pose_planner.x_lim
        ylim = pose_planner.y_lim
        zlim = pose_planner.z_lim
        xlen = xlim[1] - xlim[0]
        ylen = ylim[1] - ylim[0]
        zlen = zlim[1] - zlim[0]
        maxlen = max(xlen, ylen, zlen)
        xlim = (xlim[0] - (maxlen - xlen) / 2, xlim[1] + (maxlen - xlen) / 2)
        ylim = (ylim[0] - (maxlen - ylen) / 2, ylim[1] + (maxlen - ylen) / 2)
        zlim = (zlim[0] - (maxlen - zlen) / 2, zlim[1] + (maxlen - zlen) / 2)
        CameraPoseVisualizer(0.1 * maxlen, xlim=xlim, ylim=ylim, zlim=zlim).plot(poses)


if __name__ == "__main__":
    # main_dummy(plot=True, save_dir="scripts/drone_movements")
    save_dir = None  # "data/drone_movements"
    main(
        drone_relative=True,
        plot=True,
        save_dir=save_dir,
        n_movements=500,
        seed=1234,
        verbose=0,
    )
