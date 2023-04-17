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
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
from matplotlib import cm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation as R


class CameraPose:
    """
    Incorporates an extrinsic parameter matrix for camera pose.
    """

    def __init__(self, extrinsic_matrix: npt.NDArray[np.float_] | None = None):
        self.extrinsic_matrix = np.eye(4)
        if extrinsic_matrix is not None:
            assert extrinsic_matrix.shape == (4, 4)
            self.extrinsic_matrix[:] = extrinsic_matrix

    @staticmethod
    def from_euler(
        vector: npt.NDArray[np.float_], degrees: bool = True
    ) -> "CameraPose":
        assert vector.shape == (6,)
        pose = CameraPose()
        pose.position = vector[:3]
        pose.rotation = R.from_euler("xyz", vector[3:], degrees=degrees).as_matrix()
        return pose

    # ----------------------------------------------------------------------
    # Utils

    def as_extrinsic(self) -> npt.NDArray[np.float_]:
        return self.extrinsic_matrix

    def as_euler(self, degrees: bool = True) -> npt.NDArray[np.float_]:
        rotation = R.from_matrix(self.rotation).as_euler("xyz", degrees=degrees)
        return np.concatenate([self.position, rotation])

    def copy(self) -> "CameraPose":
        return CameraPose(self.as_extrinsic().copy())

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
    # Absolute transformations

    def translate(self, x: float = 0, y: float = 0, z: float = 0) -> "CameraPose":
        self.position = self.position + np.array([x, y, z])
        return self

    def rotate(
        self, x: float = 0, y: float = 0, z: float = 0, degrees: bool = True
    ) -> "CameraPose":
        rot = R.from_euler("xyz", [x, y, z], degrees=degrees)
        self.rotation = rot.apply(self.rotation, inverse=True)
        return self

    # ----------------------------------------------------------------------
    # Relative transformations

    def surge(self, dist: float) -> "CameraPose":
        self.position += self.rotation @ np.array([dist, 0, 0])
        return self

    def sway(self, dist: float) -> "CameraPose":
        self.position += self.rotation @ np.array([0, dist, 0])
        return self

    def heave(self, dist: float) -> "CameraPose":
        self.position += self.rotation @ np.array([0, 0, dist])
        return self

    def roll(self, angle: float, degrees: bool = True) -> "CameraPose":
        return self.rotate(x=angle, degrees=degrees)

    def pitch(self, angle: float, degrees: bool = True) -> "CameraPose":
        return self.rotate(y=angle, degrees=degrees)

    def yaw(self, angle: float, degrees: bool = True) -> "CameraPose":
        return self.rotate(z=angle, degrees=degrees)


class CameraPoseTracker:
    """
    Keeps track of the movement history of camera poses.
    """

    def __init__(self, init_pose: CameraPose) -> None:
        self.history: list[CameraPose] = [init_pose]

    def __len__(self) -> int:
        return len(self.history)

    def __iter__(self):
        self._current_index = 0
        return self

    def __next__(self):
        if self._current_index > len(self.history) - 1:
            raise StopIteration
        current_pose = self.history[self._current_index]
        self._current_index += 1
        return current_pose

    # ----------------------------------------------------------------------
    # Transformations

    def move_abs(
        self,
        trans_x: float = 0,
        trans_y: float = 0,
        trans_z: float = 0,
        rot_x: float = 0,
        rot_y: float = 0,
        rot_z: float = 0,
        degrees: bool = True,
    ) -> "CameraPoseTracker":
        """
        Movements relative to the coordinate system.

        """
        curr_pose = self.history[-1]
        next_pose = curr_pose.copy()
        # Apply transformation
        next_pose.translate(x=trans_x, y=trans_y, z=trans_z)
        next_pose.rotate(x=rot_x, y=rot_y, z=rot_z, degrees=degrees)
        # Append to history
        self.history.append(next_pose)
        return self

    def move_rel(
        self,
        surge: float = 0,
        sway: float = 0,
        heave: float = 0,
        roll: float = 0,
        pitch: float = 0,
        yaw: float = 0,
        degrees: bool = True,
    ) -> "CameraPoseTracker":
        """
        Movements relative to the camera's orientation.

        """
        curr_pose = self.history[-1]
        next_pose = curr_pose.copy()
        # Apply transformation
        next_pose.surge(surge).sway(sway).heave(heave)
        next_pose.roll(roll, degrees).pitch(pitch, degrees).yaw(yaw, degrees)
        # Append to history
        self.history.append(next_pose)
        return self

    # ----------------------------------------------------------------------
    # Export and import

    def as_pandas(self) -> pd.DataFrame:
        columns = "x[mm],y[mm],z[mm],roll_x[deg],pitch_y[deg],yaw_z[deg]".split(",")
        data = pd.DataFrame([p.as_euler() for p in self.history], columns=columns)
        return data

    @staticmethod
    def from_pandas(data: pd.DataFrame, degrees: bool = True) -> "CameraPoseTracker":
        # Pop first row
        init_row = data.iloc[0]
        data = data.iloc[1:]
        # Initialize tracker
        init_pose = CameraPose.from_euler(init_row.to_numpy(), degrees)
        poses = CameraPoseTracker(init_pose)
        # Extract all other rows
        for _, row in data.iterrows():
            poses.history.append(CameraPose.from_euler(row.to_numpy(), degrees))
        return poses

    def to_csv(self, filepath: str | Path) -> None:
        self.as_pandas().to_csv(filepath)


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
        visual_range: float = 5,
        azimuth_fov: float = 30,
        elevation_fov: float = 15,
        degrees: bool = True,
        xlim: tuple[float, float] = (-20, 20),
        ylim: tuple[float, float] = (-20, 20),
        zlim: tuple[float, float] = (-20, 20),
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
    ) -> "CameraPoseVisualizer":
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

        for p, c, a in zip(pose, color, alpha):
            self._plot(p, c, a)

        plt.title(title or "Camera Poses")
        plt.show()
        return self

    def _plot(
        self, pose: CameraPose, color: str | list[float], alpha: float
    ) -> "CameraPoseVisualizer":
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
            poses.move_rel(surge=10, yaw=30)
        else:
            poses.move_rel(surge=10, yaw=-30)

    return poses


class PlanarPosePlanner:
    """
    Planner for random camera poses with planar movements.

    Parameters
    ----------
    xlim : tuple[float, float], optional
        Left and right sway limit, by default (-20, 20)
    ylim : tuple[float, float], optional
        Forward and backward surge limit, by default (-20, 20)
    yawlim : tuple[float, float], optional
        Left and right camera angle limit, by default (-45, 45)
    max_surge : float, optional
        Maximal surge between two subsequent camera poses, by default 10
    max_sway : float, optional
        Maximal sway between two subsequent camera poses, by default 10
    max_yaw : float, optional
        Maximal yaw between two subsequent camera poses, by default 45
    degrees : bool, optional
        If True, it assumes that 'yawlim' and 'max_yaw' are in degrees, by default True
    seed : int | None, optional
        Set the initial random state for reproducability, by default None
    verbose : int, optional
        For debugging, by default 0
    """

    def __init__(
        self,
        xlim: float | tuple[float, float] = (-20, 20),
        ylim: float | tuple[float, float] = (-20, 20),
        yawlim: float | tuple[float, float] = (-45, 45),
        max_surge: float = 10,
        max_sway: float = 10,
        max_yaw: float = 30,
        degrees: bool = True,
        seed: int | None = None,
        verbose: int = 0,
    ):
        # Set camera position and angle boundaries
        if isinstance(xlim, (int, float)):
            xlim = (-np.abs(xlim), np.abs(xlim))
        if isinstance(ylim, (int, float)):
            ylim = (-np.abs(ylim), np.abs(ylim))
        if isinstance(yawlim, (int, float)):
            yawlim = (-np.abs(yawlim), np.abs(yawlim))
        self.xlim = xlim
        self.ylim = ylim
        if degrees:  # convert to radians
            yawlim = (yawlim[0] / 180 * np.pi, yawlim[1] / 180 * np.pi)
        self.yawlim = (
            (yawlim[0] + np.pi) % (2 * np.pi) - np.pi,  # normalize to [-pi, 0]
            yawlim[1] % (2 * np.pi),  # normalize to [0, pi]
        )
        # Set movability parameters
        self.max_surge = max_surge
        self.max_sway = max_sway
        self.max_yaw = (max_yaw if not degrees else max_yaw / 180 * np.pi) % (2 * np.pi)
        # Set initial state
        self.cur_x = 0.0
        self.cur_y = 0.0
        self.cur_yaw = 0.0
        # Other settings
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.verbose = verbose

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
            surge, sway, yaw = self._get_next_movement(poses)
            poses.move_rel(surge=surge, sway=sway, yaw=yaw, degrees=False)

        return poses

    def _get_next_movement(
        self, poses: CameraPoseTracker
    ) -> tuple[float, float, float]:
        # Debug print to check whether current position estimate is correct.
        # Works only if the initial pose starts at (x, y, z) = (0, 0, 0) and has
        # no initial rotations.
        self.debug("estimate:", [self.cur_x, self.cur_y, 0.0, 0.0, 0.0, self.cur_yaw])
        self.debug("true:    ", poses.history[-1].as_euler(degrees=False).tolist())

        # Get and update surge
        surge = self._get_next_surge()
        self.cur_x += surge * np.cos(self.cur_yaw)
        self.cur_y += surge * np.sin(self.cur_yaw)

        # Get and update sway
        # Note that we adjust the angle by 90 degrees (also within _get_next_sway)
        sway = self._get_next_sway()
        self.cur_x += sway * np.cos(self.cur_yaw + np.pi / 2)
        self.cur_y += sway * np.sin(self.cur_yaw + np.pi / 2)

        # Get and update yaw (make pretty with modulo)
        yaw = self._get_next_yaw()
        rel_yaw = self.cur_yaw + yaw
        self.cur_yaw = (rel_yaw + np.pi) % (2 * np.pi) - np.pi

        self.debug("next_move:", [surge, sway, yaw], "\n")
        return surge, sway, yaw

    def _get_next_yaw(self) -> float:
        # Get min/max yaw, given the min/max yaw limit
        min_yaw = (self.yawlim[0] - self.cur_yaw + np.pi) % (2 * np.pi) - np.pi
        max_yaw = (self.yawlim[1] - self.cur_yaw) % (2 * np.pi)
        # Keep yaw within the allowed min/max parameter
        min_yaw = max(-self.max_yaw, min_yaw)
        max_yaw = min(self.max_yaw, max_yaw)

        return self.rng.uniform(min_yaw, max_yaw)

    def _get_min_max_translation(
        self, min_: float, max_: float, alpha: float
    ) -> tuple[float, float]:
        """
        Get min and max translation in order not to cross the allowed border.

        Parameters
        ----------
        min_ : float
            Fixed minimum amount to move.
        max_ : float
            Fixed maximum amount to move.
        alpha : float
            Angle of the direction to move.

        Returns
        -------
        float
            Minimal amount to move (backward) in order not to cross boundaries.
        float
            Maximal amount to move (forward) in order not to cross boundaries.

        """
        # Normalize the angle alpha
        alpha = alpha % (2 * np.pi)

        # Define variables for border distances
        d_top = self.xlim[1] - self.cur_x
        d_left = self.ylim[1] - self.cur_y
        d_right = self.cur_y - self.ylim[0]
        d_bottom = self.cur_x - self.xlim[0]

        # Raise error if the point is not within the allowed boundaries
        if d_top < 0 or d_right < 0 or d_bottom < 0 or d_left < 0:
            raise AssertionError("The point is not within the allowed grid.")

        # Case 1: alpha in [0, 90]
        if 0 <= alpha <= np.pi / 2:
            self.debug("a: [0, 90]")
            max_ = min(max_, d_top / np.cos(alpha))
            max_ = min(max_, d_left / np.cos(np.pi / 2 - alpha))
            min_ = max(min_, -d_right / np.cos(np.pi / 2 - alpha))
            min_ = max(min_, -d_bottom / np.cos(alpha))

        # Case 2: alpha in [90, 180]
        elif np.pi / 2 <= alpha <= np.pi:
            self.debug("b: [90, 180]")
            min_ = max(min_, -d_top / np.cos(np.pi - alpha))
            max_ = min(max_, d_left / np.cos(np.pi / 2 - alpha))
            min_ = max(min_, -d_right / np.cos(np.pi / 2 - alpha))
            max_ = min(max_, d_bottom / np.cos(np.pi - alpha))

        # Case 3: alpha in [180, 270]
        elif np.pi <= alpha <= np.pi * 3 / 2:
            self.debug("c: [180, 270]")
            min_ = max(min_, -d_top / np.cos(np.pi - alpha))
            min_ = max(min_, d_left / np.cos(np.pi * 3 / 2 - alpha))
            max_ = min(max_, -d_right / np.cos(np.pi * 3 / 2 - alpha))
            max_ = min(max_, d_bottom / np.cos(np.pi - alpha))

        # Case 4: alpha in [270, 360]
        else:
            self.debug("d: [270, 360]")
            max_ = min(max_, d_top / np.cos(alpha))
            min_ = max(min_, d_left / np.cos(np.pi / 2 - alpha))
            max_ = min(max_, -d_right / np.cos(np.pi / 2 - alpha))
            min_ = max(min_, -d_bottom / np.cos(alpha))

        self.debug("min/max:", [min_, max_])

        return min_ * 0.99, max_ * 0.99

    def _get_next_surge(self) -> float:
        min_surge, max_surge = self._get_min_max_translation(
            -self.max_surge, self.max_surge, self.cur_yaw
        )
        return self.rng.uniform(min_surge, max_surge)

    def _get_next_sway(self) -> float:
        # Note that we adjust the angle by 90 degrees
        min_sway, max_sway = self._get_min_max_translation(
            -self.max_sway, self.max_sway, self.cur_yaw + np.pi / 2
        )
        return self.rng.uniform(min_sway, max_sway)


# ==============================================================================
# Main


def main():
    # # Dummy
    # poses = make_dummy_poses()
    # poses.to_csv(Path(__file__).parent / "dummy_pitch=0deg_roll=0deg.csv")
    # CameraPoseVisualizer().plot(poses)

    n_movements = 200
    planner_kwargs = dict(
        xlim=2000, ylim=2000, yawlim=45, max_surge=800, max_sway=800, max_yaw=30
    )
    for pitch in [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]:
        for roll in [-20, -15, -10, -5, 0, 5, 10, 15, 20]:
            seed = 1234 + (pitch * roll)

            # Make random poses
            init_pose = CameraPose().pitch(pitch).roll(roll)
            planner = PlanarPosePlanner(**planner_kwargs, seed=seed)  # type: ignore
            poses = planner.get_random_poses(init_pose, n_movements=n_movements)

            # # Visualize
            # lim = (-2000, 2000)
            # CameraPoseVisualizer(200, xlim=lim, ylim=lim, zlim=lim).plot(poses)

            # Save data
            filename = f"movements_pitch={pitch}deg_roll={roll}deg.csv"
            poses.to_csv(filename)


if __name__ == "__main__":
    main()
