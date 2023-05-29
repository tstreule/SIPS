"""
Project points from sonar (uv-)images to projection arcs in (xyz-)space.

"""
from typing import Iterable

import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R

from sips.data import CameraParams, CameraPose, SonarDatum


def _xyz_to_rtp(
    points_xyz: torch.Tensor,
    camera_pose: CameraPose | None = None,
    degrees: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Converts points from cartesian to spherical coorinate system.

    Parameters
    ----------
    points : torch.Tensor
        Points in cartesian coordinates (x, y, z) of shape (3,) or (N, 3).
    camera_pose : CameraPose | None, optional
        If given, calculates the spherical coorindates relative to the given
        camera pose, by default None
    degrees : bool, optional
        If True, returns theta and phi in degrees, by default False

    Returns
    -------
    r, theta, phi : tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Points in spherical coordinates (r, theta, phi) of shape (3,) or (N, 3).

    """
    points_xyz = torch.as_tensor(points_xyz)
    if points_xyz.ndim > 2 or points_xyz.shape[-1] != 3:
        raise ValueError("Invalid dimensions.")

    if camera_pose is not None:
        # Translate point relative to camera position
        points_xyz = points_xyz - camera_pose.position.as_tensor().to(points_xyz)
        # Rotate the shifted point by camera orientation
        rot_matrix = camera_pose.rotation.rot.inv().as_matrix()
        x, y, z = torch.as_tensor(rot_matrix).to(points_xyz) @ points_xyz.T
    else:
        x, y, z = points_xyz.T

    # Coordinate change
    r = torch.sqrt(x**2 + y**2 + z**2)
    phi = torch.arcsin(z / r)
    theta = torch.atan2(y, x)  # same as torch.arctan(y / x)
    # Handle angles
    if degrees:
        theta *= 180 / torch.pi
        phi *= 180 / torch.pi

    return r, theta, phi


def _get_pixel_delta(
    image_resolution: Iterable[int], params: CameraParams
) -> tuple[float, float]:
    width, height = image_resolution
    delta_t = params.azimuth / (width - 1)  # theta
    delta_r = (params.max_range - params.min_range) / (height - 1)  # range
    return delta_t, delta_r


def xyz_to_uv(points_xyz: torch.Tensor, sonar: SonarDatum) -> torch.Tensor:
    """
    Converts points from Euclidean (xyz) space to sonar image (uv) space.

    Parameters
    ----------
    points_xyz : torch.Tensor
        Point(s) of shape ([N,] 3) to be converted.
    sonar : SonarDatum
        Sonar information for conversion.

    Returns
    -------
    torch.Tensor
        Converted point(s) in sonar image (uv) space of shape ([N,] 2).

    """
    points_xyz = torch.as_tensor(points_xyz)
    if points_xyz.ndim > 2 or points_xyz.shape[-1] != 3:
        raise ValueError("Invalid dimensions.")

    # Convert point to spherical coordinates
    r, theta, phi = _xyz_to_rtp(points_xyz, sonar.pose, sonar.params.degrees)

    # Find step size in u (theta) and v (radius) direction
    delta_t, delta_r = _get_pixel_delta(sonar.image.shape, sonar.params)

    # Normalize angles to fit within image dimensions
    u = (sonar.params.azimuth / 2 - theta) / delta_t
    v = (r - sonar.params.min_range) / delta_r

    # Check if point is within operational range and field of view
    epsilon = 1e-6
    out_of_operatial_masks = [
        r < sonar.params.min_range - epsilon,
        r > sonar.params.max_range + epsilon,
        torch.abs(theta) > sonar.params.azimuth / 2 + epsilon,
        torch.abs(phi) > sonar.params.elevation / 2 + epsilon,
    ]
    ooo_mask = torch.stack(out_of_operatial_masks).any(0)

    # Stack pixel coordinates and filter "unseen" projections
    points_uv = torch.stack([u, v], dim=-1)
    points_uv[ooo_mask] = torch.nan

    return points_uv


def uv_to_xyz(
    points_uv: torch.Tensor, sonar: SonarDatum, n_elevations: int = 5
) -> torch.Tensor:
    """
    Samples points from sonar image (uv) space into Euclidean (xyz) space.

    Parameters
    ----------
    points_uv : torch.Tensor
        Point(s) of shape ([N,] 3) to be sampled.
    sonar : SonarDatum
        Sonar information for conversion.
    n_elevations : int, optional
        Number of samples per point, by default 5

    Returns
    -------
    torch.Tensor
        Sampled point(s) in Euclidean (xyz) space of shape ([N,] n_elevations, 2).

    """
    # Extract u / v components
    points_uv = torch.as_tensor(points_uv)
    u, v = points_uv.T

    # Find step size in u (theta) and v (radius) direction
    delta_t, delta_r = _get_pixel_delta(sonar.image.shape, sonar.params)

    # Find horizontal rotations
    theta = sonar.params.azimuth / 2 - u * delta_t
    theta_rotvec = F.pad(theta[:, None], (2, 0), value=0)  # add two 0-cols from left
    theta_rot = R.from_rotvec(theta_rotvec.cpu(), degrees=sonar.params.degrees).as_matrix()  # type: ignore
    # Find vertical rotations
    phi = (
        -torch.linspace(-1, 1, n_elevations) * (sonar.params.elevation / 2)
        if n_elevations != 1
        else torch.tensor([0])
    )
    phi_rotvec = F.pad(phi[:, None], (1, 1), value=0)  # add a 0-col from both sides
    phi_rot = R.from_rotvec(phi_rotvec.cpu(), degrees=sonar.params.degrees).as_matrix()  # type: ignore
    # Use camera pose to find direction of the sonar beam for given keypoints
    # Note: By adding new axes we find every theta_rot[i] x phi_rot[j] combination.
    rotations = sonar.pose.rotation.rot.as_matrix() @ theta_rot[:, None] @ phi_rot[None]
    rotations = torch.from_numpy(rotations).to(points_uv)

    # Find translations for given orientations/rotations
    distances = sonar.params.min_range + v * delta_r
    x_directions = F.pad(distances[:, None], (0, 2), value=0)  # add 0-cols from right
    translations = torch.einsum("ijkl,il->ijk", rotations, x_directions)

    # Find all keypoint positions given the translations
    points_xyz = sonar.pose.position.as_tensor().to(translations) + translations

    return points_xyz
