"""
Dataclasses for handling sonar data.

"""
from dataclasses import dataclass, fields
from typing import Any, Iterable, TypeVar, cast

import torch
from scipy.spatial.transform import Rotation as R

__all__ = ["CameraPose", "CameraParams", "SonarDatum", "SonarDatumPair", "SonarBatch"]

# ==============================================================================
# Utils

_T = TypeVar("_T")


def _ensure_class(value: tuple[Any, ...] | dict[str, Any] | _T, cls: type[_T]) -> _T:
    """
    Ensure that an input is of type ``cls``. Instantiate if needed.

    Parameters
    ----------
    value : tuple[Any, ...] | dict[str, Any] | _T
        Input to be checked.
    cls : type[_T]
        Required type of the ``value``.

    Returns
    -------
    _T
        Instantiated input.

    """
    if isinstance(value, tuple):
        return cls(*value)
    elif isinstance(value, dict):
        return cls(**value)
    elif not isinstance(value, cls):
        raise ValueError(f"Unsupported type '{type(value)}' - expected '{cls}'")
    return value


# ==============================================================================
# Sonar Camera Parameters


@dataclass(init=False)
class CameraPose:
    """
    Stores xyz-position and xyzw-orientation (quaternion).
    """

    position: torch.Tensor  # (3,)
    rotation: torch.Tensor  # (4,)

    def __init__(self, position: "_FloatIterable", rotation: "_FloatIterable") -> None:
        self.position = torch.as_tensor(position).float()
        assert self.position.shape == (3,)

        rotation = R.from_quat(torch.as_tensor(rotation)).as_quat()
        self.rotation = torch.from_numpy(rotation).float()
        assert self.rotation.shape == (4,)

    def as_extrinsic(self) -> torch.Tensor:
        matrix = torch.eye(4)
        matrix[:-1, -1] = self.position
        matrix[:3, :3] = torch.from_numpy(R.from_quat(self.rotation).as_matrix())
        return matrix


@dataclass
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
    degrees: bool = True

    def __post_init__(self) -> None:
        if not 0 <= self.min_range < self.max_range:
            raise ValueError("Invalid camera range")
        # Enforce degrees
        if not self.degrees:
            self.azimuth = self.azimuth / torch.pi * 180
            self.elevation = self.elevation / torch.pi * 180
            self.degrees = True


# ==============================================================================
# Sonar Data Instances


@dataclass(init=False)
class SonarDatum:
    """
    Stores sonar image, pose and camera parameters.
    """

    image: torch.Tensor  # (H,W,C)
    pose: CameraPose
    params: CameraParams
    stamp: int

    def __init__(
        self,
        image: "_FloatIterable",
        pose: "_CameraPoseLike",
        params: "_CameraParamsLike",
        stamp: int = -1,  # -1 is an invalid stamp and stands for "undefined"
    ) -> None:
        self.image = torch.as_tensor(image)
        assert self.image.ndim == 3
        self.pose = _ensure_class(pose, CameraPose)
        self.params = _ensure_class(params, CameraParams)
        self.stamp = stamp

    def __repr__(self) -> str:
        fieldreprs = [f"{f.name}=..." for f in fields(self)]
        return f"{type(self).__name__}({', '.join(fieldreprs)})"


@dataclass(init=False)
class SonarDatumPair:
    """
    Sonar datum pair for KeypointNet training.
    """

    image1: torch.Tensor  # (H,W,C)
    image2: torch.Tensor  # (H,W,C)
    pose1: CameraPose
    pose2: CameraPose
    params1: CameraParams  # TODO: (optionally) remove one param as we can not have different ones
    params2: CameraParams

    def __init__(self, sonar1: "_SonarDatumLike", sonar2: "_SonarDatumLike") -> None:
        sonar1 = _ensure_class(sonar1, SonarDatum)
        self.image1 = sonar1.image
        self.pose1 = sonar1.pose
        self.params1 = sonar1.params

        sonar2 = _ensure_class(sonar2, SonarDatum)
        self.image2 = sonar2.image
        self.pose2 = sonar2.pose
        self.params2 = sonar2.params

    def __repr__(self) -> str:
        fieldreprs = [f"{f.name}=..." for f in fields(self)]
        return f"{type(self).__name__}({', '.join(fieldreprs)})"


@dataclass(init=False)
class SonarBatch:
    image1: torch.Tensor  # (B,H,W,C)
    image2: torch.Tensor  # (B,H,W,C)
    pose1: list[CameraPose]
    pose2: list[CameraPose]
    params1: list[CameraParams]
    params2: list[CameraParams]

    def __init__(self, batch: Iterable["_SonarDatumPairLike"]) -> None:
        batch = [_ensure_class(x, SonarDatumPair) for x in batch]
        batch = cast(list[SonarDatumPair], batch)

        self.image1 = torch.stack([x.image1 for x in batch])
        self.image2 = torch.stack([x.image2 for x in batch])
        self.pose1 = [x.pose1 for x in batch]
        self.pose2 = [x.pose2 for x in batch]
        self.params1 = [x.params1 for x in batch]
        self.params2 = [x.params2 for x in batch]

    def __repr__(self) -> str:
        fieldreprs = [f"{f.name}=..." for f in fields(self)]
        return f"{type(self).__name__}({', '.join(fieldreprs)})"

    def __getitem__(self, i: int) -> SonarDatumPair:
        sonar1 = SonarDatum(self.image1[i], self.pose1[i], self.params1[i])
        sonar2 = SonarDatum(self.image2[i], self.pose2[i], self.params2[i])
        return SonarDatumPair(sonar1, sonar2)

    @property
    def batch_size(self) -> int:
        return len(self.pose1)


# ==============================================================================
# Type Hinting


_FloatIterable = Iterable[float] | torch.Tensor
_CameraPoseLike = CameraPose | tuple[_FloatIterable, _FloatIterable]
_CameraParamsLike = (
    CameraParams
    | tuple[float, float, float, float]
    | tuple[float, float, float, float, bool]
)
_SonarDatumLike = SonarDatum | tuple[_FloatIterable, _CameraPoseLike, _CameraParamsLike]
_SonarDatumPairLike = SonarDatumPair | tuple[_SonarDatumLike, _SonarDatumLike]
