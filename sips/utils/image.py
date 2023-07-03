# Copyright 2020 Toyota Research Institute.  All rights reserved.

from functools import lru_cache

import torch


@lru_cache(maxsize=None)
def meshgrid(
    B: int, H: int, W: int, dtype: torch.dtype, device: str, normalized: bool = False
):
    """Create mesh-grid given batch size, height and width dimensions.

    Parameters
    ----------
    B: int
        Batch size
    H: int
        Grid Height
    W: int
        Batch size
    dtype: torch.dtype
        Tensor dtype
    device: str
        Tensor device
    normalized: bool
        Normalized image coordinates or integer-grid.

    Returns
    -------
    xs: torch.Tensor
        Batched mesh-grid x-coordinates (BHW).
    ys: torch.Tensor
        Batched mesh-grid y-coordinates (BHW).
    """
    if normalized:
        xs = torch.linspace(-1, 1, W, device=device, dtype=dtype)
        ys = torch.linspace(-1, 1, H, device=device, dtype=dtype)
    else:
        xs = torch.linspace(0, W - 1, W, device=device, dtype=dtype)
        ys = torch.linspace(0, H - 1, H, device=device, dtype=dtype)
    ys, xs = torch.meshgrid([ys, xs], indexing="ij")
    return xs.repeat([B, 1, 1]), ys.repeat([B, 1, 1])


@lru_cache(maxsize=None)
def image_grid(
    B: int,
    H: int,
    W: int,
    dtype: torch.dtype,
    device: torch.device,
    ones: bool = True,
    normalized: bool = False,
) -> torch.Tensor:
    """Create an image mesh grid with shape B3HW given image shape BHW

    Parameters
    ----------
    B: int
        Batch size
    H: int
        Grid Height
    W: int
        Batch size
    dtype: str
        Tensor dtype
    device: str
        Tensor device
    ones : bool
        Use (x, y, 1) coordinates
    normalized: bool
        Normalized image coordinates or integer-grid.

    Returns
    -------
    grid: torch.Tensor
        Mesh-grid for the corresponding image shape (B3HW)
    """
    xs, ys = meshgrid(B, H, W, dtype, device, normalized=normalized)
    coords = [xs, ys]
    if ones:
        coords.append(torch.ones_like(xs))  # BHW
    grid = torch.stack(coords, dim=1)  # B3HW
    return grid


def normalize_sonar(images: torch.Tensor) -> torch.Tensor:
    assert images.ndim == 4
    return images.div(255 / 2).sub(1)


def normalize_2d_coordinate(
    coord: torch.Tensor, height: int, width: int
) -> torch.Tensor:
    """
    Normalize 2D coordinates

    """
    _, d, _, _ = coord.shape
    assert d == 2

    coord[:, 0] = (coord[:, 0] / ((height - 1) * 0.5)) - 1
    coord[:, 1] = (coord[:, 1] / ((width - 1) * 0.5)) - 1
    return coord.permute(0, 2, 3, 1)


def unnormalize_2d_coordinate(
    coord: torch.Tensor, height: int, width: int
) -> torch.Tensor:
    """
    Unnormalize 2D coordinates

    """
    _, _, _, d = coord.shape
    assert d == 2

    coord[..., 0] = (coord[..., 0] + 1) * (0.5 * (height - 1))
    coord[..., 1] = (coord[..., 1] + 1) * (0.5 * (width - 1))
    return coord.permute(0, 3, 1, 2)
