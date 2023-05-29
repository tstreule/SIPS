from pathlib import Path
from time import time

import typer
from typing_extensions import Annotated

import drone_movements as dm

app = typer.Typer()


@app.callback(invoke_without_command=True)
def main(
    # Initialization
    init_roll: float = 0,
    init_pitch: float = 0,
    init_yaw: float = 0,
    # Planning
    radius_start: float = 0.3,
    radius_stop: float = 1,
    max_theta: float = 45,
    max_phi: float = 30,
    n_radius_steps: int = 5,
    n_theta_steps: int = 5,
    n_phi_steps: int = 5,
    roi_angle: float = 30,
    cam2drone: float = 0,
    degrees: Annotated[bool, typer.Option("--degrees/--radians")] = True,
    # Plotting
    plot: bool = True,
    # Export
    csv_export: bool = False,
    csv_filename: str = "example",
    csv_sep: str = "\t",
) -> None:

    # Initialize
    init_pose = dm.MovableCameraPose.neutral_pose()
    init_pose.rotate([init_roll, init_pitch, init_yaw], degrees=degrees)
    tracker = dm.CameraPoseTracker(init_pose)

    # Plan poses
    tracker = dm.plan_roi_poses_3d(
        # fmt: off
        radius_start, radius_stop, max_theta, max_phi,
        n_radius_steps, n_theta_steps, n_phi_steps, roi_angle,
        tracker=tracker, degrees=degrees,
    )

    # Adjust camera view to drone view
    tracker = dm.camera2drone(tracker, angle=cam2drone, degrees=degrees)

    # Plotting
    if plot:
        dm.plot_camera_poses(tracker, show=True)

    # Export
    if csv_export:
        filename = f"{csv_filename}_{time() * 1e3:.0f}.csv"
        filepath = Path("data/drone_movements", filename)
        filepath.mkdir(parents=True, exist_ok=True)
        tracker.to_csv(filepath, sep=csv_sep)


if __name__ == "__main__":
    app()
