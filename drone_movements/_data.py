"""
Dataclass extensions required for drone movement planning.

"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.spatial.transform import Rotation as R
from typing_extensions import Self

from sips.dataclasses import CameraPose, CameraPosition, CameraRotation

__all__ = ["Movement", "MovableCameraPose", "CameraPoseTracker"]


@dataclass
class Movement:
    """
    Incorporates a relative movement (translation and rotation).
    """

    surge: float
    sway: float
    heave: float
    roll: float
    pitch: float
    yaw: float
    degrees: bool = True

    def __post_init__(self) -> None:
        if self.degrees:
            return

        # Enforce degrees
        self.roll *= 180 / np.pi
        self.pitch *= 180 / np.pi
        self.yaw *= 180 / np.pi
        self.degrees = True

    def as_dict(self, units: bool = True, length_unit: str = "m") -> dict[str, float]:
        lu = au = ""
        if units:
            lu = f"[{length_unit}]"  # length unit
            au = "[deg]"  # angle unit

        translation = {
            f"surge{lu}": self.surge,
            f"sway{lu}": self.sway,
            f"heave{lu}": self.heave,
        }
        rotation = {
            f"roll{au}": self.roll,
            f"pitch{au}": self.pitch,
            f"yaw{au}": self.yaw,
        }
        return {**translation, **rotation}


class MovableCameraPose(CameraPose):
    """
    Incorporates camera position and orientation and is movable with relative movements.
    """

    @classmethod
    def neutral_pose(cls) -> "MovableCameraPose":
        return cls((0, 0, 0), (0, 0, 0, 1))

    def translate(self, dists: npt.ArrayLike) -> None:
        dists = np.asarray(dists)
        assert dists.shape == (3,), "Invalid dimensions"
        x, y, z = dists
        self.position.x += x
        self.position.y += y
        self.position.z += z

    def surge(self, dist: float) -> Self:
        self.translate(self.rotation.rot.apply([dist, 0, 0]))
        return self

    def sway(self, dist: float) -> Self:
        self.translate(self.rotation.rot.apply([0, dist, 0]))
        return self

    def heave(self, dist: float) -> Self:
        self.translate(self.rotation.rot.apply([0, 0, dist]))
        return self

    def rotate(self, angles: npt.ArrayLike, degrees: bool = True) -> None:
        angles = np.asarray(angles)
        rot = R.from_rotvec(angles, degrees=degrees)  # type: ignore
        new_quat = (self.rotation.rot * rot).as_quat()
        self.rotation = CameraRotation(*new_quat)

    def roll(self, angle: float, degrees: bool = True) -> Self:
        self.rotate([angle, 0, 0], degrees=degrees)
        return self

    def pitch(self, angle: float, degrees: bool = True) -> Self:
        self.rotate([0, angle, 0], degrees=degrees)
        return self

    def yaw(self, angle: float, degrees: bool = True) -> Self:
        self.rotate([0, 0, angle], degrees=degrees)
        return self

    def copy(self) -> "MovableCameraPose":
        position = CameraPosition(*self.position.as_array())
        rotation = CameraRotation(*self.rotation.rot.as_quat())
        return MovableCameraPose(position, rotation)


class CameraPoseTracker:
    """
    Keeps track of the movement history of camera poses.

    Parameters
    ----------
    init_pose : MovableCameraPose | None
        Initial pose. If None, a neutral pose will be used.
    """

    def __init__(self, init_pose: MovableCameraPose | None = None) -> None:
        if init_pose is None:
            init_pose = MovableCameraPose.neutral_pose()

        self.abs_history: list[MovableCameraPose] = [init_pose]
        self.rel_history: list[Movement] = [Movement(0, 0, 0, 0, 0, 0)]

    def __len__(self) -> int:
        assert len(self.abs_history) == len(self.rel_history)
        return len(self.abs_history)

    def __getitem__(self, ix: int) -> tuple[MovableCameraPose, Movement]:
        return self.abs_history[ix], self.rel_history[ix]

    def __iter__(self) -> Self:
        self._current_index = 0
        return self

    def __next__(self) -> tuple[MovableCameraPose, Movement]:
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
        self.rel_history.append(Movement(surge, sway, heave, roll, pitch, yaw, degrees))
        return self

    # ----------------------------------------------------------------------
    # Utils

    def get_pos_limits(self) -> tuple[tuple[float, float], ...]:
        xyz_arr = []
        for pose in self.abs_history:
            p = pose.position
            xyz_arr.append([p.x, p.y, p.z])

        xyz_mins = np.array(xyz_arr).min(0)
        xyz_maxs = np.array(xyz_arr).max(0)

        return tuple(zip(xyz_mins, xyz_maxs))

    # ----------------------------------------------------------------------
    # Export

    def to_pandas(self, add_units: bool = True) -> pd.DataFrame:
        df = pd.DataFrame([p.as_dict(units=add_units) for p in self.rel_history])
        return df

    def to_csv(
        self,
        filepath: str | Path,
        sep: str = ",",
        add_units: bool = True,
    ) -> None:
        self.to_pandas(add_units=add_units).to_csv(filepath, index=False, sep=sep)
