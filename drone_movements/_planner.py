"""
Tool for planning 3d movements of sonar data.

"""
import math
from typing import Literal

import numpy as np

from drone_movements._data import CameraPoseTracker

__all__ = ["plan_roi_poses_2d", "plan_roi_poses_3d", "camera2drone"]


# ==============================================================================
# Planning


def _yaw_region_of_interest(
    tracker: CameraPoseTracker, theta: float, n_views: int = 3
) -> None:
    # Go to first yaw extreme
    tracker.move(yaw=-theta, degrees=False)
    # Do all views
    theta_delta = 2 * theta / (n_views - 1)
    for _ in range(n_views - 1):
        tracker.move(yaw=theta_delta, degrees=False)
    # Reset orientation
    tracker.move(yaw=-theta, degrees=False)


def _move_along_circle(
    rotate: Literal["pitch", "yaw"],
    tracker: CameraPoseTracker,
    theta: float,
    radius: float,
    max_distance: float = math.inf,
    yaw_roi_theta: float | None = None,
    yaw_roi_n_views: int | None = None,
) -> None:
    # Get chord length (straight line, contrary to the arc)
    chord_length = 2 * radius * math.sin(theta / 2)

    # Move twice with half angle to satisfy max_distance
    if chord_length > max_distance:
        n_steps = int(np.ceil(chord_length / max_distance))
        for _ in range(n_steps):
            _move_along_circle(
                # fmt: off
                rotate, tracker, theta / n_steps, radius, max_distance,
                yaw_roi_theta, yaw_roi_n_views,
            )
        return

    # Read off translation and rotation direction
    if rotate == "pitch":
        translate = "heave"
    elif rotate == "yaw":
        translate = "sway"
    else:
        raise ValueError("Unknown argument")

    # Normally align with the tangent line passing through the desired point
    tracker.move(**{rotate: theta / 2}, degrees=False)  # type: ignore[misc]
    # Move sideways perpendicular to your orientation by a distance equal to the chord length
    tracker.move(**{translate: -chord_length}, degrees=False)
    # Look towards the center of the circle again
    tracker.move(**{rotate: theta / 2}, degrees=False)  # type: ignore[misc]

    # Look around
    if yaw_roi_theta is not None:
        if yaw_roi_n_views is not None:
            _yaw_region_of_interest(tracker, yaw_roi_theta, yaw_roi_n_views)
        else:
            _yaw_region_of_interest(tracker, yaw_roi_theta)


def plan_roi_poses_2d(
    radius_start: float = 0.3,
    radius_stop: float = 1,
    max_theta: float = 45,
    n_radius_steps: int = 5,
    n_theta_steps: int = 5,
    roi_angle: float = 30,
    max_theta_distance: float | Literal["auto"] = "auto",
    tracker: CameraPoseTracker | None = None,
    back_to_init_pose: bool = False,
    degrees: bool = True,
) -> CameraPoseTracker:

    # Convert to radian
    max_theta = max_theta / 180 * math.pi if degrees else max_theta
    roi_angle = roi_angle / 180 * math.pi if degrees else roi_angle

    if max_theta_distance == "auto":
        max_theta_distance = radius_stop / 3

    # Get step size for radius and angle
    delta_theta = 2 * max_theta / (n_theta_steps - 1)
    delta_radius = (radius_stop - radius_start) / (n_radius_steps - 1)

    # Initialize camera pose tracker and current relative positions
    tracker = CameraPoseTracker() if tracker is None else tracker
    cur_radius = radius_start
    cur_theta = -max_theta
    # Move camera to the first desired position
    _move_along_circle("yaw", tracker, cur_theta, cur_radius, max_theta_distance)

    # Move along circles of different radius
    for r_step in range(n_radius_steps):

        left_to_right = r_step % 2 == 0
        _yaw_region_of_interest(tracker, roi_angle)
        for _ in range(n_theta_steps - 1):
            mv_theta = delta_theta if left_to_right else -delta_theta
            cur_theta += mv_theta
            _move_along_circle(
                "yaw", tracker, mv_theta, cur_radius, max_theta_distance, roi_angle
            )

        # Break loop
        if r_step > n_radius_steps - 2:
            break

        # Move to next circle line
        cur_radius += delta_radius
        tracker.move(surge=-delta_radius)

    if back_to_init_pose:
        _move_along_circle("yaw", tracker, -cur_theta, cur_radius)
        tracker.move(surge=radius_stop - radius_start)

    return tracker


def plan_roi_poses_3d(
    radius_start: float = 0.3,
    radius_stop: float = 1,
    max_theta: float = 45,
    max_phi: float = 30,
    n_radius_steps: int = 5,
    n_theta_steps: int = 5,
    n_phi_steps: int = 5,
    roi_angle: float = 30,
    max_theta_distance: float | Literal["auto"] = "auto",
    tracker: CameraPoseTracker | None = None,
    back_to_init_pose: bool = False,
    degrees: bool = True,
):

    # Convert phi angle to radian and get its step size
    max_phi = max_phi / 180 * math.pi if degrees else max_phi
    delta_phi = 2 * max_phi / (n_phi_steps - 1)

    # Initialize camera pose tracker and current relative phi
    tracker = CameraPoseTracker() if tracker is None else tracker
    cur_phi = -max_phi
    # Move camera to the first desired plane
    _move_along_circle("pitch", tracker, cur_phi, -radius_start)

    # Move along planes on different phi
    for phi_step in range(n_phi_steps):
        plan_roi_poses_2d(
            radius_start,
            radius_stop,
            max_theta,
            n_radius_steps,
            n_theta_steps,
            roi_angle,
            max_theta_distance,
            tracker=tracker,
            back_to_init_pose=True,  # important to properly ``_move_along_circle``!
            degrees=degrees,
        )

        # Break loop
        if phi_step > n_phi_steps - 2:
            break

        # Move to next plane
        cur_phi += delta_phi
        _move_along_circle("pitch", tracker, delta_phi, -radius_start)

    if back_to_init_pose:
        _move_along_circle("pitch", tracker, -cur_phi, -radius_start)

    return tracker


# ==============================================================================
# Utils


def camera2drone(
    camera_tracker: CameraPoseTracker, angle: float = 0, degrees: bool = True
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
        Whether ``angle`` is given in degrees or radians, by default True

    Returns
    -------
    CameraPoseTracker
        Converted poses.

    """
    # Camera and drone tracker are equal
    if angle == 0:
        return camera_tracker

    # Set up new pose tracker while adjusting the initial orientation
    init_drone = camera_tracker.abs_history[0].copy().pitch(-angle, degrees=degrees)
    drone_tracker = CameraPoseTracker(init_pose=init_drone)

    # Translate the relative camera movements to relative drone movements
    # Note: We skip i=0 since initial pose is already set up
    for i in range(1, len(camera_tracker)):

        # Get previous camera and drone orientation
        prev_camera_orient = camera_tracker.abs_history[i - 1].rotation.rot
        prev_drone_orient = drone_tracker.abs_history[i - 1].rotation.rot

        # Find relative drone movement
        delta_camera = camera_tracker.rel_history[i]
        camera_mov = [getattr(delta_camera, val) for val in ("surge", "sway", "heave")]
        drone_mov = (prev_drone_orient.inv() * prev_camera_orient).apply(camera_mov)

        # Find relative drone rotation
        next_drone_orient = (
            camera_tracker.abs_history[i].copy().pitch(-angle, degrees=degrees)
        ).rotation.rot
        drone_rot = (next_drone_orient.inv() * prev_drone_orient).inv()

        # Apply
        # Note: Big 'XYZ' correspond to intrinsic rotations in given sequential order
        surge, sway, heave = drone_mov
        roll, pitch, yaw = drone_rot.as_euler("XYZ", degrees=True)
        drone_tracker.move(surge, sway, heave, roll, pitch, yaw, degrees=True)

        # Assert equal camera and drone position
        np.testing.assert_allclose(
            drone_tracker.abs_history[i].position.as_array(),
            camera_tracker.abs_history[i].position.as_array(),
            rtol=1e-15,
            atol=1e-14,
        )

    return drone_tracker
