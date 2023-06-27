"""
Functions to create sonar dummy data.

"""
import torch
from scipy.spatial.transform import Rotation as R

from sips.data import CameraParams, SonarBatch, SonarDatum, SonarDatumPair

__all__ = [
    "SONAR_TUPLE",
    "get_random_datum",
    "get_random_datum_pair",
    "get_random_batch",
]


# Make dummy image and pose
# Note that you can easily make a new pose e.g. by:
#  import drone_movements as dm
#  pose = dm.CameraPose()
#  track = dm.CameraPoseTracker(pose)
#  track.move(surge=4, sway=3, heave=2, yaw=-30, pitch=20)
#  xyzlim = (-10, 10)
#  dm.CameraPoseVisualizer(10, 60, 15, xlim=xyzlim, ylim=xyzlim, zlim=xyzlim).plot(track)
#  for p in track.abs_history:
#      print(p.position.tolist(), p.get_quat().tolist())
_IMAGE = torch.zeros(1, 512, 512)
_PARAMS = CameraParams(min_range=1, max_range=10, azimuth=130, elevation=20)
_POSE_1 = ((0, 0, 0), (0, 0, 0, 1))
_POSE_2 = ((1, 3, 2), (-0.044943455, 0.167731259, -0.254887002, 0.951251242))
SONAR_TUPLE = SonarDatumPair((_IMAGE, _POSE_1, _PARAMS), (_IMAGE, _POSE_2, _PARAMS))


def get_random_datum(seed: int | None = None) -> SonarDatum:
    if seed is not None:
        raise NotImplementedError

    # Make random image
    image = torch.rand_like(_IMAGE, dtype=torch.float) * 255
    image = image.to(torch.uint8)

    # Make random pose
    position = torch.normal(torch.zeros(3), 1)  # meters
    xyz_rotation = torch.normal(torch.zeros(3), 5)  # degrees
    rotation = R.from_euler("xyz", xyz_rotation, degrees=True).as_quat()

    return SonarDatum(image, (position, rotation), _PARAMS)


def get_random_datum_pair(seed: int | None = None) -> SonarDatumPair:
    seed2 = seed + 1 if seed is not None else None
    return SonarDatumPair(get_random_datum(seed=seed), get_random_datum(seed=seed2))


def get_random_batch(batch_size: int = 8, seed: int | None = None) -> SonarBatch:
    pairs = [
        get_random_datum_pair(seed + 2 * i if seed else None) for i in range(batch_size)
    ]
    return SonarBatch(pairs)
