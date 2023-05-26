"""
Small module to plan drone movements for acquiring data.

"""
from drone_movements._data import CameraPoseTracker, MovableCameraPose, Movement
from drone_movements._planner import camera2drone, plan_roi_poses_2d, plan_roi_poses_3d
from drone_movements._plotting import CameraPoseVisualizer, plot_camera_poses

__all__ = [
    # fmt: off
    "Movement", "MovableCameraPose", "CameraPoseTracker",
    "plan_roi_poses_2d", "plan_roi_poses_3d", "camera2drone",
    "CameraPoseVisualizer", "plot_camera_poses",
]
