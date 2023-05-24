from typing import Iterable

import torch
import torch.nn.functional as F


def _unravel_index_2d(
    flat_index: torch.Tensor, shape: Iterable[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    # Similar to ``numpy.unravel_index`` but can only handle 2D shape

    _, h = shape
    row, col = flat_index // h, flat_index % h

    return row, col


def _torch_lexsort(keys: torch.Tensor, dim: int = -1) -> torch.Tensor:
    # Credits: https://discuss.pytorch.org/t/numpy-lexsort-equivalent-in-pytorch/47850/4

    if keys.ndim < 2:
        raise ValueError(f"keys must be at least 2 dimensional, but {keys.ndim=}.")
    if len(keys) == 0:
        raise ValueError(f"Must have at least 1 key, but {len(keys)=}.")

    idx = keys[0].argsort(dim=dim, stable=True)
    for k in keys[1:]:
        idx = idx.gather(dim, k.gather(dim, idx).argsort(dim=dim, stable=True))

    return idx


def _groupwise_smallest_values_mask(
    groups: torch.Tensor, values: torch.Tensor
) -> torch.Tensor:
    h, w = groups.shape
    device = groups.device

    # Combine groups, values, and positional index into a single 2D array
    pos_index = torch.arange(h)
    gv_indexed = torch.column_stack((groups.cpu(), values.cpu(), pos_index))

    # Sort BA_indexed based on the first and second columns (indices 0 and 1)
    gv_indexed_sorted = gv_indexed[_torch_lexsort(gv_indexed[:, :w].T)]

    # Find unique tuples in B and their corresponding indices
    unique_B, indices = torch.unique_consecutive(
        gv_indexed_sorted[:, :w], return_inverse=True, dim=0
    )
    # Group the data based on unique tuples in B
    split_indices = torch.arange(1, h)[(indices[1:] - indices[:-1]) == 1]
    grouped_data = torch.tensor_split(gv_indexed_sorted, split_indices.tolist())

    # Find the smallest value in A for each unique tuple in B
    idx_smallest_values = torch.tensor(
        [group[torch.argmin(group[:, w]), w + 1] for group in grouped_data],
    ).to(torch.int64)

    # Create the filter mask using positional index
    filter_mask = torch.zeros(h, dtype=torch.bool)
    filter_mask[idx_smallest_values] = True

    return filter_mask.to(device)


def match_keypoints_2d(
    kp1_uv: torch.Tensor,
    kp2_uv_proj: torch.Tensor,
    convolution_size: int,
    window_size: int = 3,
    distance_threshold: float = 12,
):
    # NOTE: It is assumed that ``kp1_uv`` is ordered according to a meshgrid. Otherwize,
    #  the concept of "rounding" (convert to int) does not represent a index.

    assert kp1_uv.device == kp2_uv_proj.device
    _device = kp1_uv.device

    kp1_uv = torch.as_tensor(kp1_uv)
    assert kp1_uv.ndim == 3
    assert kp1_uv.shape[0] == kp1_uv.shape[1] and kp1_uv.shape[2] == 2
    kp2_uv_proj = torch.as_tensor(kp2_uv_proj)

    PAD = (window_size - 1) // 2
    window_shape = (window_size, window_size)

    # Mask out points that don't project onto the other image
    kp2_mask = torch.isfinite(kp2_uv_proj).any(-1)  # same as .all(-1)

    # Rounding down gives the index of the other images point
    idx = (kp2_uv_proj[kp2_mask] / convolution_size).to(torch.int64)  # shape: (H, 2)
    # idx = idx.flip(1)  # first index row, then the column

    # Create a sliding window view such that we can compare each projected keypoint
    # with the (by rounding) closest keypoint and its neighbors in the other image
    kp1_uv_padded = F.pad(kp1_uv, (0, 0, PAD, PAD, PAD, PAD), value=torch.nan)
    #      padded shape: (H+2*PAD, W+2*PAD, 2)
    kp1_uv_strided = kp1_uv_padded.unfold(0, window_size, 1).unfold(1, window_size, 1)
    #      strided shape: (H, W, 2, window_size, window_size)

    # Calculate the distances
    masked_diffs = kp2_uv_proj[kp2_mask][..., None, None] - kp1_uv_strided[tuple(idx.T)]
    masked_dists = torch.linalg.vector_norm(masked_diffs, dim=1)
    masked_dists = masked_dists.nan_to_num(torch.inf).reshape(-1, window_size**2)
    distances = torch.full(
        (*kp2_mask.shape, window_size**2), torch.inf, device=_device
    )
    distances[kp2_mask] = masked_dists

    # Find the closest points
    # Flatten such that we get a unique argmin for every set of projected keypoints in 2
    flat_dists = distances.flatten(start_dim=1)
    flat_dists_min, flat_dists_argmin = torch.min(flat_dists, dim=1)

    # Note to me:
    #  I want to unravel the index two times.
    #  1) to get the index for the ``distances`` variable
    #  2) to get the index for the ``kp1_uv`` variable
    row, col = _unravel_index_2d(flat_dists_argmin, distances.shape[1:])
    kp2_uv_match = kp2_uv_proj.take_along_dim(row[:, None, None], 1).squeeze(1)
    idx_kp1 = (kp2_uv_match / convolution_size).to(torch.int64)  # .flip(1)
    idx_kp1_rel = torch.stack(_unravel_index_2d(col, window_shape), 1) - 1
    # clipping due to bad keypoints that didn't get projected onto the other image
    idx_kp1 = (idx_kp1 + idx_kp1_rel).clip(0)
    # --> ``idx_kp1`` is the position of the closest keypoint for kp2_uv_projected

    mask_closest = _groupwise_smallest_values_mask(idx_kp1, flat_dists_min)

    # Remove matches with distance greater than threshold
    mask_closest[mask_closest.clone()] = (
        flat_dists_min[mask_closest.clone()] < distance_threshold
    )
    # >> SELECT ``kp1_uv``      WITH ``idx_kp1[mask_closest]`` AS INDEX
    # >> SELECT ``kp2_uv_proj`` WITH ``...`` AS INDEX

    kp1_uv_match = kp1_uv[tuple(idx_kp1.T)]

    return kp1_uv_match[mask_closest], kp2_uv_match[mask_closest]
