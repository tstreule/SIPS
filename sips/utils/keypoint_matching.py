from typing import Iterable, Literal

import torch
import torch.nn.functional as F

# ==============================================================================
# Utils


def _unravel_index_2d(
    flat_index: torch.Tensor, shape: Iterable[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    # Similar to ``numpy.unravel_index`` but can only handle 2D shape

    _, h = shape
    row, col = flat_index // h, flat_index % h

    return row, col


def _unravel_index_3d(
    flat_index: torch.Tensor, shape: Iterable[int]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Similar to ``numpy.unravel_index`` but can only handle 3D shape

    w, h, d = shape
    a, sub_flat_index = _unravel_index_2d(flat_index, (w, h * d))
    b, c = _unravel_index_2d(sub_flat_index, (h, d))

    return a, b, c


def _torch_lexsort(keys: torch.Tensor, dim: int = -1) -> torch.Tensor:
    # Credits: https://discuss.pytorch.org/t/numpy-lexsort-equivalent-in-pytorch/47850/5

    if keys.ndim < 2:
        raise ValueError(f"keys must be at least 2 dimensional, but {keys.ndim=}.")
    if len(keys) == 0:
        raise ValueError(f"Must have at least 1 key, but {len(keys)=}.")

    # for-loop is required due to bugs mentioned in credits
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
    pos_index = torch.arange(h, device=device)
    gv_indexed = torch.column_stack((groups, values, pos_index))

    # Sort gv_indexed based on the first and second columns (indices 0 and 1)
    gv_indexed_sorted = gv_indexed[_torch_lexsort(gv_indexed[:, :w].T)]

    # Find unique group tuples and their corresponding indices
    _, indices = torch.unique_consecutive(
        gv_indexed_sorted[:, :w], return_inverse=True, dim=0
    )
    # Group the data based on unique group tuples
    split_indices = torch.arange(1, h, device=device)[(indices[1:] - indices[:-1]) == 1]
    grouped_data_tup = torch.tensor_split(gv_indexed_sorted, split_indices.tolist())

    # Put the group again into one tensor
    max_group_len = int(torch.tensor([len(group) for group in grouped_data_tup]).max())
    grouped_data = torch.stack(
        [
            F.pad(group, (0, 0, 0, max_group_len - len(group)), value=torch.inf)
            for group in grouped_data_tup
        ]
    )

    # Find the smallest value for each unique group tuple
    argmins = torch.argmin(grouped_data[:, :, w], dim=1, keepdim=True)
    idx_smallest_values = grouped_data[..., w + 1].gather(1, argmins).flatten().int()

    # Create the filter mask using positional index
    filter_mask = torch.zeros(h, dtype=torch.bool, device=device)
    filter_mask[idx_smallest_values] = True

    return filter_mask


def _project_point_on_linesegment(
    v: torch.Tensor, w: torch.Tensor, p: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    # Partial credits: https://stackoverflow.com/a/1501725
    assert v.ndim == w.ndim == p.ndim

    # Put dimension at last place
    if dim not in (-1, v.ndim - 1):
        v = v.movedim(dim, -1)
        w = w.movedim(dim, -1)
        p = p.movedim(dim, -1)

    # Consider the line extending the segment, parameterized as v + t (w - v).
    # We find projection of point p onto the line.
    # It falls where t = [(p-v) . (w-v)] / |w-v|^2
    # We clamp t from [0,1] to handle points outside the segment vw.
    l2 = torch.sum((w - v) ** 2, dim=-1)  # length squared -> |w-v|^2
    t = (torch.einsum("...i,...i->...", p - v, w - v) / l2).clamp(0, 1)  # dot product
    projection = v + t[..., None] * (w - v)

    # Handle v == w case
    mask = l2 == 0
    if mask.any():
        mask = torch.broadcast_to(mask, projection.shape[:-1])
        projection[mask] = v[mask]

    # Put dimension back to original place
    if dim not in (-1, v.ndim - 1):
        projection = projection.movedim(-1, dim)

    return projection


def _point2point_distance(
    p1: torch.Tensor, p2: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    return torch.sum((p1 - p2) ** 2, dim=dim) ** 0.5


def _point2linesegment_distance(
    v: torch.Tensor, w: torch.Tensor, p: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """
    Return minimum distance between line segment vw and point p.

    """
    projection = _project_point_on_linesegment(v, w, p, dim=dim)
    return _point2point_distance(p, projection, dim=dim)


# ==============================================================================
# Keypoint matching


def _p2p_match_keypoints_2d(
    kp1_uv: torch.Tensor,
    kp2_uv_proj: torch.Tensor,
    convolution_size: int,
    distance_threshold: float = 0.5,
    window_size: int = 3,
) -> tuple[torch.Tensor, torch.Tensor]:
    # NOTE: It is assumed that ``kp1_uv`` is ordered according to a meshgrid. Otherwize,
    #  the concept of "rounding" (convert to int) does not represent a index.

    # Check input
    C1, _, _ = kp1_uv.shape
    D, C2, H, W = kp2_uv_proj.shape
    assert C1 == C2 == 2

    kp1_uv = kp1_uv.permute(1, 2, 0)  # H, W, C
    kp2_uv_proj = kp2_uv_proj.view(D, C2, H * W).permute(2, 0, 1)  # (D, C, H*W)

    assert kp1_uv.device == kp2_uv_proj.device
    device = kp1_uv.device

    # Set padding and window size
    PAD = (window_size - 1) // 2
    window_shape = (window_size, window_size)

    # Mask out points that don't project onto the other image
    kp2_mask = torch.isfinite(kp2_uv_proj).any(-1)  # same as .all(-1)

    # Rounding down gives the index of the other images point
    idx = (kp2_uv_proj[kp2_mask] / convolution_size).to(torch.int64)  # shape: (H, 2)
    idx[..., 0] = idx[..., 0].clip(0, H - 1)
    idx[..., 1] = idx[..., 1].clip(0, W - 1)

    # Create a sliding window view such that we can compare each projected keypoint
    # with the (by rounding) closest keypoint and its neighbors in the other image
    kp1_uv_padded = F.pad(kp1_uv, (0, 0, PAD, PAD, PAD, PAD), value=torch.nan)
    #      padded shape: (H+2*PAD, W+2*PAD, 2)
    kp1_uv_strided = kp1_uv_padded.unfold(0, window_size, 1).unfold(1, window_size, 1)
    #      strided shape: (H, W, 2, window_size, window_size)

    # Calculate the distances
    masked_dists = _point2point_distance(
        kp2_uv_proj[kp2_mask][..., None, None], kp1_uv_strided[tuple(idx.T)], dim=1
    )
    masked_dists = masked_dists.nan_to_num(torch.inf).reshape(-1, window_size**2)
    distances = torch.full(
        (*kp2_mask.shape, window_size**2), torch.inf, device=device
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
    idx_kp1 = torch.nan_to_num((kp2_uv_match / convolution_size), 0).to(torch.int64)
    idx_kp1_rel = torch.stack(_unravel_index_2d(col, window_shape), 1) - 1
    # clipping due to bad keypoints that didn't get projected onto the other image
    idx_kp1 = (idx_kp1 + idx_kp1_rel).clip(0)
    # --> ``idx_kp1`` is the position of the closest keypoint for kp2_uv_projected

    # Find the closest keypoints per given index
    # while ignoring distances greater than threshold
    mask_threshold = flat_dists_min < convolution_size * distance_threshold
    mask_closest = torch.zeros_like(mask_threshold)
    mask_closest[mask_threshold] = _groupwise_smallest_values_mask(
        idx_kp1[mask_threshold], flat_dists_min[mask_threshold]
    )

    kp1_uv_match = kp1_uv[tuple(idx_kp1.T)]
    return kp1_uv_match[mask_closest], kp2_uv_match[mask_closest]


def _p2l_match_keypoints_2d(
    kp1_uv: torch.Tensor,
    kp2_uv_proj: torch.Tensor,
    convolution_size: int,
    distance_threshold: float = 0.5,
    window_size: int = 3,
) -> tuple[torch.Tensor, torch.Tensor]:
    # NOTE: It is assumed that ``kp1_uv`` is ordered according to a meshgrid. Otherwize,
    #  the concept of "rounding" (convert to int) does not represent a index.

    # Check input
    C1, _, _ = kp1_uv.shape
    D, C2, H, W = kp2_uv_proj.shape
    assert C1 == C2 == 2

    kp1_uv = kp1_uv.permute(1, 2, 0)  # (H, W, C)
    kp2_uv_proj = kp2_uv_proj.view(D, C2, H * W).permute(2, 0, 1)  # (D, C, H*W)

    assert kp1_uv.device == kp2_uv_proj.device
    device = kp1_uv.device

    # Set padding and window size
    PAD = (window_size - 1) // 2
    window_shape = (window_size, window_size)

    # Mask out points that don't project onto the other image
    kp2_mask = torch.isfinite(kp2_uv_proj).any(-1).any(1)  # same as .all(-1).any(1)

    # Rounding down gives the index of the other images point
    idx = torch.nan_to_num(kp2_uv_proj[kp2_mask] / convolution_size).int()  # (-1, D, 2)
    idx[..., 0] = idx[..., 0].clip(0, H - 1)
    idx[..., 1] = idx[..., 1].clip(0, W - 1)

    # Create a sliding window view such that we can compare each projected keypoint
    # with the (by rounding) closest keypoint and its neighbors in the other image
    kp1_uv_padded = F.pad(kp1_uv, (0, 0, PAD, PAD, PAD, PAD), value=torch.nan)
    #      padded shape: (H+2*PAD, W+2*PAD, 2)
    kp1_uv_strided = kp1_uv_padded.unfold(0, window_size, 1).unfold(1, window_size, 1)
    #      strided shape: (H, W, 2, window_size, window_size)

    # Calculate the distances
    line_points = kp2_uv_proj[kp2_mask]
    points = kp1_uv_strided[tuple(idx.flatten(0, 1).T)].unflatten(0, idx.shape[:2])
    points = points.flatten(-2, -1)
    masked_distances = _point2linesegment_distance(
        line_points[:, :-1, None, :, None],
        line_points[:, 1:, None, :, None],
        points[:, None],
        dim=3,
    )
    distances = torch.full(
        (*kp2_mask.shape, *masked_distances.shape[1:]), torch.inf, device=device
    )
    distances[kp2_mask] = masked_distances.nan_to_num(torch.inf)

    # Find the closest points
    # Flatten such that we get a unique argmin for every set of projected keypoints in 2
    flat_dists = distances.flatten(1)
    flat_dists_min, flat_dists_argmin = flat_dists.min(1)

    # Note to me:
    #  I want to unravel the index two times.
    #  1) to get the index for the ``distances`` variable
    #  2) to get the index for the ``kp1_uv`` variable
    line, row, col = _unravel_index_3d(flat_dists_argmin, distances.shape[1:])
    kp2_uv_match = kp2_uv_proj.take_along_dim(row[:, None, None], 1).squeeze(1)
    idx_kp1 = torch.nan_to_num(kp2_uv_match / convolution_size).int()
    idx_kp1_rel = torch.stack(_unravel_index_2d(col, window_shape), 0).T - 1
    # clipping due to bad keypoints that didn't get projected onto the other image
    idx_kp1 = idx_kp1 + idx_kp1_rel
    idx_kp1[..., 0] = idx_kp1[..., 0].clip(0, H - 1)
    idx_kp1[..., 1] = idx_kp1[..., 1].clip(0, W - 1)
    # --> ``idx_kp1`` is the position of the closest keypoint for kp2_uv_projected
    kp1_uv_match = kp1_uv[tuple(idx_kp1.T)]
    kp2_uv_match = kp2_uv_proj.take_along_dim(line[:, None, None], 1).squeeze(1)
    kp2_uv_match_2 = kp2_uv_proj.take_along_dim(line[:, None, None] + 1, 1).squeeze(1)

    # Find the closest keypoints per given index
    # while ignoring distances greater than threshold
    mask_threshold = flat_dists_min < convolution_size * distance_threshold
    mask_closest = torch.zeros_like(mask_threshold)
    mask_closest[mask_threshold] = _groupwise_smallest_values_mask(
        idx_kp1[mask_threshold], flat_dists_min[mask_threshold]
    )

    p = kp1_uv_match[mask_closest]
    a = kp2_uv_match[mask_closest]
    b = kp2_uv_match_2[mask_closest]
    r = _project_point_on_linesegment(a, b, p)

    return p, r


def match_keypoints_2d(
    kp1_uv: torch.Tensor,
    kp2_uv_proj: torch.Tensor,
    convolution_size: int,
    distance_threshold: float = 0.5,
    window_size: int = 3,
    distance: Literal["p2p", "p2l"] = "p2l",
) -> tuple[torch.Tensor, ...]:
    # Use point to point distance (faster)
    if distance == "p2p":
        return _p2p_match_keypoints_2d(
            kp1_uv, kp2_uv_proj, convolution_size, distance_threshold, window_size
        )
    # Use point to line distance (less precise)
    elif distance == "p2l":
        return _p2l_match_keypoints_2d(
            kp1_uv, kp2_uv_proj, convolution_size, distance_threshold, window_size
        )
    # Error
    else:
        raise ValueError(f"Invalid {distance=}")


def batch_match_keypoints_2d(
    kp1_uv: torch.Tensor,
    kp2_uv_proj: torch.Tensor,
    convolution_size: int,
    distance_threshold: float = 0.5,
    window_size: int = 3,
    distance: Literal["p2p", "p2l"] = "p2l",
) -> list[tuple[torch.Tensor, ...]]:
    batch_size = kp1_uv.shape[0]

    # Apply for every batch index
    kp_uv_matches: list[tuple[torch.Tensor, ...]] = []
    for b in range(batch_size):
        kp_uv_match = match_keypoints_2d(
            kp1_uv[b],
            kp2_uv_proj[b],
            convolution_size,
            distance_threshold,
            window_size,
            distance=distance,
        )
        kp_uv_matches.append(kp_uv_match)

    return kp_uv_matches
