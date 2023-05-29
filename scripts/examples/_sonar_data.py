import torch
from scipy.spatial.transform import Rotation as R

from sips.data import CameraParams, SonarDatum, SonarDatumTuple

__all__ = ["SONAR_TUPLE", "get_random_datum", "get_random_datum_tuple"]


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
_IMAGE = torch.zeros(512, 512)
_PARAMS = CameraParams(min_range=1, max_range=10, azimuth=130, elevation=20)
_POSE_1 = ((0, 0, 0), (0, 0, 0, 1))
_POSE_2 = ((1, 3, 2), (-0.044943455, 0.167731259, -0.254887002, 0.951251242))
SONAR_TUPLE = SonarDatumTuple((_IMAGE, _POSE_1, _PARAMS), (_IMAGE, _POSE_2, _PARAMS))


def get_random_datum(seed: int | None = None) -> SonarDatum:

    if seed is not None:
        raise NotImplementedError

    # Make random image
    image = torch.normal(_IMAGE * 0)

    # Make random pose
    position = torch.normal(torch.zeros(3), 1)  # meters
    xyz_rotation = torch.normal(torch.zeros(3), 5)  # degrees
    rotation = R.from_euler("xyz", xyz_rotation, degrees=True).as_quat()

    return SonarDatum(image, (position, rotation), _PARAMS)


def get_random_datum_tuple(seed: int | None = None) -> SonarDatumTuple:
    seed2 = seed + 1 if seed is not None else None
    return SonarDatumTuple(get_random_datum(seed=seed), get_random_datum(seed=seed2))
