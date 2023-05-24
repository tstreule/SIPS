import torch

from sips.dataclasses import CameraParams, SonarDatum, SonarDatumTuple

__all__ = ["SONAR_TUPLE", "get_random_datum", "get_random_datum_tuple"]


# TODO: Delete this dummy example as soon as "get_random_datum" function implemented
# Make dummy image and pose
_IMAGE = torch.zeros(512, 512)
_PARAMS = CameraParams(min_range=1, max_range=10, azimuth=130, elevation=20)
_POSE_1 = ((0, 0, 0), (0, 0, 0, 1))
_POSE_2 = ((1, 3, 2), (-0.044943455, 0.167731259, -0.254887002, 0.951251242))
SONAR_TUPLE = SonarDatumTuple((_IMAGE, _POSE_1, _PARAMS), (_IMAGE, _POSE_2, _PARAMS))


def get_random_datum(seed: int | None = None) -> SonarDatum:
    # TODO: Randomly make dummy image and pose
    #  Note that you can easily make a new pose e.g. by:
    #  from scripts.drone_movements.generate import *
    #  pose = CameraPose()
    #  track = CameraPoseTracker(pose)
    #  track.move(surge=4, sway=3, heave=2, yaw=-30, pitch=20)
    #  xyzlim = (-10, 10)
    #  CameraPoseVisualizer(10, 60, 15, xlim=xyzlim, ylim=xyzlim, zlim=xyzlim).plot(track)
    #  for p in track.abs_history:
    #      print(p.position.tolist(), p.get_quat().tolist())
    raise NotImplementedError


def get_random_datum_tuple(seed: int | None = None) -> SonarDatumTuple:
    print("WARNING: Returning static SonarDatumTuple -> not yet implemented...")
    return SONAR_TUPLE
    # TODO: The first datum maybe always should be centered and with initial orientation
    return SonarDatumTuple(get_random_datum(seed=seed), get_random_datum(seed=None))
