"""
Project points from sonar (uv-)images to projection arcs in (xyz-)space.

"""
from typing import Iterable

import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R

from sips.data import CameraParams, CameraPose

__all__ = [
    "uv_to_xyz",
    "uv_to_xyz_batch",
    "xyz_to_uv",
    "xyz_to_uv_batch",
    "warp_image",
    "warp_image_batch",
]

# ==============================================================================
# Utils


def _get_pixel_delta(
    image_resolution: Iterable[int], params: CameraParams
) -> tuple[float, float]:
    """
    Find step size in u (theta) and v (radius) direction

    """
    width, height = image_resolution
    delta_t = params.azimuth / (width - 1)  # theta
    delta_r = (params.max_range - params.min_range) / (height - 1)  # range
    return delta_t, delta_r


# ==============================================================================
# Batched coordinate transforms


def uv_to_xyz_batch(
    points_uv: torch.Tensor,
    params: list[CameraParams],
    pose: list[CameraPose],
    image_resolution: Iterable[int],
    n_elevations: int = 5,
) -> torch.Tensor:
    """
    Samples points from sonar image (uv) space into cartesian (xyz) space.

    Parameters
    ----------
    points_uv : torch.Tensor
        Points to be sampled. Shape: (B,2,H,W).
    params : list[CameraParams]
        Camera parameters for reconstruction.
    pose : list[CameraPose]
        Camera poses for reconstruction.
    image_resolution : Iterable[int]
        Image resolution of the sonar (uv) image.
    n_elevations : int, optional
        Number of samples per uv-point, by default 5

    Returns
    -------
    torch.Tensor
        Sampled points in cartesian space. Shape: (B,n_elevations,3,H,W)

    """
    # Input check
    B, C, H, W = points_uv.shape
    assert C == 2

    # Extract u / v components
    u, v = uv = points_uv.view(B, 2, H * W).permute(1, 0, 2)

    # Find step size in u (theta) and v (radius) direction
    deltas = torch.tensor([_get_pixel_delta(image_resolution, p) for p in params])
    delta_t = deltas[:, 0, None].to(uv)
    delta_r = deltas[:, 1, None].to(uv)

    # Check whether degrees or radians
    degrees = params[0].degrees
    assert all(degrees == p.degrees for p in params)

    # Make params tensors for convenience
    azimuth = torch.tensor([p.azimuth for p in params]).to(uv)
    min_range = torch.tensor([p.min_range for p in params]).to(uv)

    # Find horizontal rotations
    theta = azimuth[:, None] / 2 - u * delta_t
    theta_rotvec = F.pad(theta[..., None], (2, 0), value=0)  # add two 0-cols from left
    theta_rot_flat = R.from_rotvec(theta_rotvec.flatten(0, 1).detach().cpu(), degrees=degrees).as_matrix()  # type: ignore
    theta_rot = torch.from_numpy(theta_rot_flat).view(B, H * W, 3, 3).to(uv)
    # Find vertical rotations
    phi = (
        -torch.linspace(-1, 1, n_elevations) if n_elevations > 1 else torch.tensor([0])
    ) * torch.tensor([p.elevation / 2 for p in params])[:, None]
    phi_rotvec = F.pad(phi[..., None], (1, 1), value=0)  # add a 0-col from both sides
    phi_rot_flat = R.from_rotvec(phi_rotvec.flatten(0, 1).cpu(), degrees=degrees).as_matrix()  # type: ignore
    phi_rot = torch.from_numpy(phi_rot_flat).view(B, n_elevations, 3, 3).to(uv)
    # Use camera pose to find direction of the sonar beam for given keypoints
    # Note: By adding new axes we find every theta_rot[i] x phi_rot[j] combination.
    cam_pose = torch.from_numpy(
        R.from_quat(torch.stack([p.rotation for p in pose]).cpu()).as_matrix()
    ).to(uv)
    rotations = cam_pose[:, None, None] @ theta_rot[:, :, None] @ phi_rot[:, None]

    # Find translations for given orientations/rotations
    distances = min_range[:, None] + v * delta_r
    x_directions = F.pad(distances[..., None], (0, 2), value=0)  # add 0-cols from right
    translations = torch.einsum("bijkl,bil->bijk", rotations, x_directions.to(uv))

    # Find all keypoint positions given the translations
    position = torch.stack([p.position for p in pose]).to(translations)
    points_xyz = position[:, None, None] + translations

    return points_xyz.permute(0, 2, 3, 1).view(B, n_elevations, 3, H, W)


def _xyz_to_rtp_batch(
    points_xyz: torch.Tensor,
    camera_pose: list[CameraPose] | None = None,
    degrees: bool = False,
) -> torch.Tensor:
    """
    Converts points form cartesian (xyz) to spherical (rtp) system.

    Parameters
    ----------
    points_xyz : torch.Tensor
        Cartesian coordinates. Shape: (B,3,H)
    camera_pose : list[CameraPose] | None, optional
        If given, calculates the spherical coordinates relative to
        the given camera pose, by default None
    degrees : bool, optional
        If True, returns theta and phi in degrees, by default False

    Returns
    -------
    torch.Tensor
        Points in spherical coordinates (r, theta, phi). Shape: (B,3,H)

    """
    # Input check
    _, C, _ = points_xyz.shape
    assert C == 3

    # Get xyz components
    if camera_pose is not None:
        # Translate point relative to camera position
        positions = torch.stack([cp.position for cp in camera_pose]).to(points_xyz)
        points_xyz = points_xyz - positions[..., None]
        # Rotate the shifted point by camera orientation
        rot_quats = torch.stack([cp.rotation for cp in camera_pose])
        rot_matrix = R.from_quat(rot_quats.cpu()).inv().as_matrix()
        points_xyz = torch.einsum(
            "bij,bjk->bik", torch.from_numpy(rot_matrix).to(points_xyz), points_xyz
        )
    x, y, z = points_xyz.permute(1, 0, 2)

    # Coordinate change
    r = torch.sqrt(x**2 + y**2 + z**2)
    phi = torch.arcsin(z / r)
    theta = torch.atan2(y, x)  # same as torch.arctan(y / x)
    # Handle angles
    if degrees:
        theta *= 180 / torch.pi
        phi *= 180 / torch.pi

    return torch.stack([r, theta, phi], dim=1)


def xyz_to_uv_batch(
    points_xyz: torch.Tensor,
    params: list[CameraParams],
    pose: list[CameraPose],
    image_resolution: Iterable[int],
) -> torch.Tensor:
    """
    Converts points from cartesian (xyz) space into sonar image (uv) space.

    Parameters
    ----------
    points_xyz : torch.Tensor
        Points to be converted. Shape: (B,N,3,H,W) or (B,3,H,W)
    params : list[CameraParams]
        Camera parameters for projection.
    pose : list[CameraPose]
        Camera poses for projection.
    image_resolution : Iterable[int]
        Image resolution of the sonar (uv) image.

    Returns
    -------
    torch.Tensor
        Projected points in sonar image space. Shape: (B,N,3,H,W) or (B,3,H,W)

    """
    if points_xyz.ndim == 5:
        B, N_ELEVATIONS, C, H, W = points_xyz.shape
        points_xyz_flat = points_xyz.permute(0, 2, 3, 4, 1).flatten(-2, -1)
        output = xyz_to_uv_batch(points_xyz_flat, params, pose, image_resolution)
        return output.unflatten(-1, (W, N_ELEVATIONS)).permute(0, 4, 1, 2, 3)

    # Input check
    B, C, H, W = points_xyz.shape
    assert C == 3

    # Check whether degrees or radians
    degrees = params[0].degrees
    assert all(degrees == p.degrees for p in params)

    # Flatten and convert to spherical coordinates
    points_xyz_flat = points_xyz.view(B, 3, H * W)
    r, theta, phi = _xyz_to_rtp_batch(points_xyz_flat, pose, degrees).permute(1, 0, 2)

    # Find step size in u (theta) and v (radius) direction
    deltas = torch.tensor([_get_pixel_delta(image_resolution, p) for p in params])
    delta_t = deltas[:, 0, None].to(points_xyz)
    delta_r = deltas[:, 1, None].to(points_xyz)

    # Make params tensors for convenience
    min_range = torch.tensor([p.min_range for p in params]).to(points_xyz)
    max_range = torch.tensor([p.max_range for p in params]).to(points_xyz)
    azimuth = torch.tensor([p.azimuth for p in params]).to(points_xyz)
    elevation = torch.tensor([p.elevation for p in params]).to(points_xyz)

    # Normalize angles to fit within image dimensions
    u = ((azimuth / 2)[:, None] - theta) / delta_t
    v = (r - min_range[:, None]) / delta_r

    # Check if point is within operational range and field of view
    epsilon = 1e-5
    out_of_operatial_masks = [
        r.lt(min_range[:, None] - epsilon),
        r.gt(max_range[:, None] + epsilon),
        torch.abs(theta).gt(azimuth[:, None] / 2 + epsilon),
        torch.abs(phi).gt(elevation[:, None] / 2 + epsilon),
    ]
    ooo_mask = torch.stack(out_of_operatial_masks).any(0)

    # Stack pixel coordinates and filter "unseen" projections
    points_uv = torch.stack([u, v])
    points_uv[:, ooo_mask] = torch.nan

    return points_uv.permute(1, 0, 2).view(B, 2, H, W)


def warp_image_batch(
    points_uv: torch.Tensor,
    source_params: list[CameraParams],
    source_pose: list[CameraPose],
    target_params: list[CameraParams],
    target_pose: list[CameraPose],
    image_resolution: Iterable[int],
    n_elevations: int = 5,
) -> torch.Tensor:
    """
    Warp a sonar image batch into another image space (arc projection).

    Parameters
    ----------
    points_uv : torch.Tensor
        Points to be projected. Shape: (B,2,H,W)
    source_params : list[CameraParams]
        Source camera parameters.
    source_pose : list[CameraPose]
        Source camera poses.
    target_params : list[CameraParams]
        Target camera parameters.
    target_pose : list[CameraPose]
        Target camera poses.
    image_resolution : Iterable[int]
        Image resolution of a sonar (uv) image.
    n_elevations : int, optional
        Number of samples per uv-point, by default 5

    Returns
    -------
    torch.Tensor
        Arc projected image points. Shape: (B,n_elevations,2,H,W)

    """
    points_xyz = uv_to_xyz_batch(
        points_uv, source_params, source_pose, image_resolution, n_elevations
    )
    points_uv_proj = xyz_to_uv_batch(
        points_xyz, target_params, target_pose, image_resolution
    )
    return points_uv_proj


# ==============================================================================
# Non-batched Coordinate transforms


def uv_to_xyz(
    points_uv: torch.Tensor,
    params: CameraParams,
    pose: CameraPose,
    image_resolution: Iterable[int],
    n_elevations: int = 5,
) -> torch.Tensor:
    """
    Samples points from sonar image (uv) space into cartesian (xyz) space.

    """
    _, _, _ = points_uv.shape  # checks dim
    batched_out = uv_to_xyz_batch(
        points_uv.unsqueeze(0), [params], [pose], image_resolution, n_elevations
    )
    return batched_out.squeeze(0)


def _xyz_to_rtp(
    points_xyz: torch.Tensor,
    camera_pose: CameraPose | None = None,
    degrees: bool = False,
) -> torch.Tensor:
    """
    Converts points from cartesian to spherical coorinate system.

    """
    _, _, _ = points_xyz.shape  # checks dim
    camera_pose_ = [camera_pose] if camera_pose else None
    batched_out = _xyz_to_rtp_batch(points_xyz.unsqueeze(0), camera_pose_, degrees)
    return batched_out.squeeze(0)


def xyz_to_uv(
    points_xyz: torch.Tensor,
    params: CameraParams,
    pose: CameraPose,
    image_resolution: Iterable[int],
) -> torch.Tensor:
    """
    Converts points from Euclidean (xyz) space to sonar image (uv) space.

    """
    batched_out = xyz_to_uv_batch(
        points_xyz.unsqueeze(0), [params], [pose], image_resolution
    )
    return batched_out.squeeze(0)


def warp_image(
    points_uv: torch.Tensor,
    from_params: CameraParams,
    from_pose: CameraPose,
    to_params: CameraParams,
    to_pose: CameraPose,
    image_resolution: Iterable[int],
    n_elevations: int = 5,
) -> torch.Tensor:
    """
    Warp a sonar image into another image space (arc projection).

    """
    points_xyz = uv_to_xyz(
        points_uv, from_params, from_pose, image_resolution, n_elevations
    )
    points_uv_proj = xyz_to_uv(points_xyz, to_params, to_pose, image_resolution)
    return points_uv_proj
