"""
Alternative to scipy's `Rotation` class for converting rotation representations
as scipy unfortunately does not support torch Tensors on GPU devices.

"""
import torch


def normalize_quat(quaternion: torch.Tensor) -> torch.Tensor:
    """
    Normalize rotation quaternions.

    Args:
        quaternion (torch.Tensor): Rotation quaternions of shape (batch_size, 4).

    Returns:
        torch.Tensor:  Normalized rotation quaternions of shape (batch_size, 4).

    """
    if not torch.is_floating_point(quaternion):
        quaternion = quaternion.float()

    quaternion_norm = torch.norm(quaternion, p=2, dim=1, keepdim=True)
    normalized_quaternion = quaternion / quaternion_norm

    return normalized_quaternion


def matrix_to_quat(rotation_matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrices to rotation quaternions.

    Args:
        rotation_matrix (torch.Tensor): Rotation matrices of shape (batch_size, 3, 3).

    Returns:
        torch.Tensor: Rotation quaternions of shape (batch_size, 4).

    """
    batch_size = rotation_matrix.size(0)

    # Extract rotation matrix elements
    r11, r12, r13 = (
        rotation_matrix[:, 0, 0],
        rotation_matrix[:, 0, 1],
        rotation_matrix[:, 0, 2],
    )
    r21, r22, r23 = (
        rotation_matrix[:, 1, 0],
        rotation_matrix[:, 1, 1],
        rotation_matrix[:, 1, 2],
    )
    r31, r32, r33 = (
        rotation_matrix[:, 2, 0],
        rotation_matrix[:, 2, 1],
        rotation_matrix[:, 2, 2],
    )

    # Compute quaternion components
    trace = r11 + r22 + r33
    qw = torch.sqrt(1 + trace) / 2
    qx = (r32 - r23) / (4 * qw)
    qy = (r13 - r31) / (4 * qw)
    qz = (r21 - r12) / (4 * qw)

    # Concatenate quaternion components
    quaternion = torch.stack((qx, qy, qz, qw), dim=1)
    quaternion = normalize_quat(quaternion)

    return quaternion.view(batch_size, 4)


def quat_to_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation quaternions to rotation matrices.

    Args:
        quaternion (torch.Tensor): Rotation quaternions of shape (batch_size, 4).

    Returns:
        torch.Tensor: Rotation matrices of shape (batch_size, 3, 3).

    """
    quaternion = normalize_quat(quaternion)
    batch_size = quaternion.size(0)

    # Extract quaternion components
    x, y, z, w = quaternion[:, 0], quaternion[:, 1], quaternion[:, 2], quaternion[:, 3]

    # Normalize quaternion
    quaternion_norm = torch.norm(quaternion, p=2, dim=1)
    x_normalized, y_normalized, z_normalized, w_normalized = (
        x / quaternion_norm,
        y / quaternion_norm,
        z / quaternion_norm,
        w / quaternion_norm,
    )

    # Compute rotation matrix elements
    xx = x_normalized * x_normalized
    xy = x_normalized * y_normalized
    xz = x_normalized * z_normalized
    yy = y_normalized * y_normalized
    yz = y_normalized * z_normalized
    zz = z_normalized * z_normalized
    wx = w_normalized * x_normalized
    wy = w_normalized * y_normalized
    wz = w_normalized * z_normalized

    # Create rotation matrix
    rotation_matrix = torch.zeros((batch_size, 3, 3), dtype=x.dtype, device=x.device)
    rotation_matrix[:, 0, 0] = 1 - 2 * (yy + zz)
    rotation_matrix[:, 0, 1] = 2 * (xy - wz)
    rotation_matrix[:, 0, 2] = 2 * (xz + wy)
    rotation_matrix[:, 1, 0] = 2 * (xy + wz)
    rotation_matrix[:, 1, 1] = 1 - 2 * (xx + zz)
    rotation_matrix[:, 1, 2] = 2 * (yz - wx)
    rotation_matrix[:, 2, 0] = 2 * (xz - wy)
    rotation_matrix[:, 2, 1] = 2 * (yz + wx)
    rotation_matrix[:, 2, 2] = 1 - 2 * (xx + yy)

    return rotation_matrix.view(batch_size, 3, 3)


def rotvec_to_matrix(
    rotation_vector: torch.Tensor, degrees: bool = False
) -> torch.Tensor:
    """
    Converts rotation vectors to rotation matrices.

    Args:
        rotation_vector (torch.Tensor): Rotation vectors of shape (batch_size, 3).
        degrees (bool): Whether the rotation vector is in degrees or radians.
            Defaults to False (radians).

    Returns:
        torch.Tensor: Rotation matrices of shape (batch_size, 3, 3).

    """
    if degrees:
        rotation_vector = rotation_vector * (torch.pi / 180.0)

    batch_size = rotation_vector.size(0)
    theta = torch.norm(rotation_vector, dim=1, keepdim=True)
    unit_rotation_vector = rotation_vector / (theta + 1e-6)

    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)

    a = unit_rotation_vector[:, 0].unsqueeze(1)
    b = unit_rotation_vector[:, 1].unsqueeze(1)
    c = unit_rotation_vector[:, 2].unsqueeze(1)

    rotation_matrix = torch.stack(
        [
            a * a * (1 - cos_theta) + cos_theta,
            a * b * (1 - cos_theta) - c * sin_theta,
            a * c * (1 - cos_theta) + b * sin_theta,
            b * a * (1 - cos_theta) + c * sin_theta,
            b * b * (1 - cos_theta) + cos_theta,
            b * c * (1 - cos_theta) - a * sin_theta,
            c * a * (1 - cos_theta) - b * sin_theta,
            c * b * (1 - cos_theta) + a * sin_theta,
            c * c * (1 - cos_theta) + cos_theta,
        ],
        dim=1,
    )

    return rotation_matrix.view(batch_size, 3, 3)
