"""
Dataclasses for handling sonar data.

"""
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from scipy.spatial.transform import Rotation as R

__all__ = [
    "CameraPosition",
    "CameraRotation",
    "CameraPose",
    "CameraParams",
    "SonarDatumTuple",
]

# ==============================================================================
# Sonar Camera Parameters


@dataclass
class CameraPosition:
    """
    Stores xyz-position.
    """

    x: float
    y: float
    z: float

    def as_array(self) -> npt.NDArray[np.float64]:
        """Array representation of the position."""
        return np.array([self.x, self.y, self.z])

    def as_tensor(self) -> torch.Tensor:
        """Array representation of the position."""
        return torch.tensor([self.x, self.y, self.z])


@dataclass
class CameraRotation:
    """
    Stores xyzw-orientation (quaternion).
    """

    x: float
    y: float
    z: float
    w: float

    def __post_init__(self) -> None:
        quat = np.array([self.x, self.y, self.z, self.w])
        self.rot = R.from_quat(quat)
        # Safety check
        if not np.allclose(self.rot.as_quat(), quat):
            raise ValueError("Invalid camera rotation quaternion")


@dataclass(init=False)
class CameraPose:
    """
    Stores xyz-position and xyzw-orientation (quaternion).
    """

    position: CameraPosition
    rotation: CameraRotation

    def __init__(
        self, position: "_CameraPositionLike", rotation: "_CameraRotationLike"
    ) -> None:

        # Set position
        if isinstance(position, tuple):
            self.position = CameraPosition(*position)
        elif isinstance(position, dict):
            self.position = CameraPosition(**position)
        elif not isinstance(position, CameraPosition):
            raise ValueError("Unsupported position dtype.")
        else:
            self.position = position

        # Set rotation
        if isinstance(rotation, tuple):
            self.rotation = CameraRotation(*rotation)
        elif isinstance(rotation, dict):
            self.rotation = CameraRotation(**rotation)
        elif not isinstance(rotation, CameraRotation):
            raise ValueError("Unsupported rotation dtype.")
        else:
            self.rotation = rotation


@dataclass(init=False)
class CameraParams:
    """
    Stores sonar camera parameters.

    Notes
    -----
    ``self.degrees`` is always ``True`` since it will be stored in degrees even if
    radians were given.
    """

    min_range: float
    max_range: float
    azimuth: float
    elevation: float

    def __init__(
        self,
        min_range: float,
        max_range: float,
        azimuth: float,
        elevation: float,
        degrees: bool = True,
    ) -> None:

        if not 0 <= min_range < max_range:
            raise ValueError("Invalid camera range")
        self.min_range = min_range
        self.max_range = max_range

        if not degrees:  # enforce degrees
            azimuth = azimuth / math.pi * 180
            elevation = elevation / math.pi * 180
        self.azimuth = azimuth
        self.elevation = elevation

    @property
    def degrees(self) -> bool:  # note that degrees are enforced above
        """``True`` if the stored angles are given in degrees."""
        return True


# ---
# TYPE HINTING
_CameraPositionLike = CameraPosition | dict[str, float] | tuple[float, float, float]
_CameraRotationLike = (
    CameraRotation | dict[str, float] | tuple[float, float, float, float]
)
_CameraPoseLike = (
    CameraPose | dict[str, Any] | tuple[_CameraPositionLike, _CameraRotationLike]
)
_CameraParamsLike = CameraParams | dict[str, Any] | tuple[float, float, float, float]


# ==============================================================================
# Sonar Data Tuple (for training)


@dataclass(init=False)
class SonarDatum:
    """
    Sonar datum containing camera image, camera pose and camera parameters.
    """

    image: torch.Tensor
    pose: CameraPose
    params: CameraParams

    def __init__(
        self, image: torch.Tensor, pose: _CameraPoseLike, params: _CameraParamsLike
    ) -> None:

        # Set camera image
        self.image = torch.as_tensor(image)

        # Set camera pose
        if isinstance(pose, tuple):
            pose = CameraPose(*pose)
        elif isinstance(pose, dict):
            pose = CameraPose(**pose)
        elif not isinstance(pose, CameraPose):
            raise ValueError("Invalid camera pose type")
        self.pose = pose

        # Set camera params
        if isinstance(params, tuple):
            params = CameraParams(*params)
        elif isinstance(params, dict):
            params = CameraParams(**params)
        elif not isinstance(params, CameraParams):
            raise ValueError("Invalid camera params type")
        self.params = params


@dataclass(init=False)
class SonarDatumTuple:
    """
    Sonar datum tuple for KeypointNet training.
    """

    sonar1: SonarDatum = field(repr=False)
    sonar2: SonarDatum = field(repr=False)

    def __init__(self, sonar1: "_SonarDatumLike", sonar2: "_SonarDatumLike") -> None:

        for i, sonar in enumerate((sonar1, sonar2)):
            if isinstance(sonar, tuple):
                sonar = SonarDatum(*sonar)
            elif isinstance(sonar, dict):
                sonar = SonarDatum(**sonar)
            elif not isinstance(sonar, SonarDatum):
                raise ValueError(f"Invaid sonar{i+1} datum type")

            setattr(self, f"sonar{i+1}", sonar)


# ---
# TYPE HINTING
_SonarDatumLike = (
    SonarDatum
    | dict[str, Any]
    | tuple[torch.Tensor, _CameraPoseLike, _CameraParamsLike]
)
_SonarDatumTupleLike = (
    SonarDatumTuple
    | dict[str, _SonarDatumLike]
    | tuple[_SonarDatumLike, _SonarDatumLike]
)
