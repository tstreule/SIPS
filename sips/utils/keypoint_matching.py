from typing import Iterable, Literal

import torch
import torch.nn.functional as F

__all__ = ["match_keypoints_2d", "batch_match_keypoints_2d"]


# ==============================================================================
# Utils


def _unravel_index_2d(
    flat_index: torch.Tensor, shape: Iterable[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Similar to ``numpy.unravel_index`` but can only handle 2D shape

    """
    _, h = shape
    row, col = flat_index // h, flat_index % h
    return row, col


def _unravel_index_3d(
    flat_index: torch.Tensor, shape: Iterable[int]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Similar to ``numpy.unravel_index`` but can only handle 3D shape

    """
    w, h, d = shape
    a, sub_flat_index = _unravel_index_2d(flat_index, (w, h * d))
    b, c = _unravel_index_2d(sub_flat_index, (h, d))
    return a, b, c


def _torch_lexsort(keys: torch.Tensor, dim: int = -1) -> torch.Tensor:
    # Credits: https://discuss.pytorch.org/t/numpy-lexsort-equivalent-in-pytorch/47850/5
    """
    numpy.lexsort equivalent function in pytorch

    """
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
    """
    Creates a mask that is True for the smallest value in each group.

    """
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


# ==============================================================================
# Distance: Point-to-Point and Point-to-LineSegment


def _project_point_on_linesegment(
    v: torch.Tensor, w: torch.Tensor, p: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    # Partial credits: https://stackoverflow.com/a/1501725
    """
    Projects 2D points ``p`` along dimension ``dim`` on line segments ``v``-``w``.

    """
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
    """
    Calculates the Point-to-Point distances between ``p1`` and ``p2``.

    """
    return torch.sum((p1 - p2) ** 2, dim=dim) ** 0.5


def _point2linesegment_distance(
    v: torch.Tensor, w: torch.Tensor, p: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """
    Calculates the Point-to-LineSegment distances.

    Every point ``p`` is projected onto the line segments ``v``-``w``.

    """
    projection = _project_point_on_linesegment(v, w, p, dim=dim)
    return _point2point_distance(p, projection, dim=dim)


# ==============================================================================
# Keypoint matching


def _batch_point2point_match(
    kp1_uv_strided: torch.Tensor, kp2_uv_proj: torch.Tensor, convolution_size: int
) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """
    Using Point-to-Point distance metric, finds
    1) min distances
    2) indices of the closest keypoints for kp2_uv_projected
    3) kp1-kp2 keypoint matches (non-filtered)

    """
    # Check input
    B, H, W, C, WS, WS = kp1_uv_strided.shape
    B, H__W, D, C = kp2_uv_proj.shape
    assert H__W == H * W
    device = kp1_uv_strided.device
    assert kp1_uv_strided.device == kp2_uv_proj.device

    # Mask out points that don't project onto the other image
    kp2_mask = torch.isfinite(kp2_uv_proj).any(-1)  # (B,H*W,D)
    # Rounding down gives the index of the other images point
    idx = torch.nan_to_num(kp2_uv_proj[kp2_mask] / convolution_size).int()  # (-1,2)
    idx[..., 0] = idx[..., 0].clip(0, H - 1)
    idx[..., 1] = idx[..., 1].clip(0, W - 1)

    # Calculate the distances
    batch_idx = kp2_mask.int() * torch.arange(B, device=device)[:, None, None]
    batch_idx = batch_idx[kp2_mask]
    masked_dists = _point2point_distance(
        kp2_uv_proj[kp2_mask][..., None, None],
        kp1_uv_strided.movedim(0, 2)[tuple(idx.T)]
        .movedim(1, 0)
        .take_along_dim(batch_idx[None, :, None, None, None], 0)
        .squeeze(0),
        dim=1,
    )
    masked_dists = masked_dists.nan_to_num(torch.inf).view(-1, WS**2)
    distances = torch.full((*kp2_mask.shape, WS**2), torch.inf, device=device)
    distances[kp2_mask] = masked_dists

    # Find the closest points
    # Flatten such that we get a unique argmin for every set of projected keypoints in 2
    flat_dists = distances.flatten(2)
    flat_dists_min, flat_dists_argmin = flat_dists.min(2)

    # Unravel to get index for the ``distances`` variable
    #  - ``row`` and ``col``: are required to reconstruct the index in kp1
    row, col = _unravel_index_2d(flat_dists_argmin, distances.shape[-2:])

    # Select kp2 matches
    kp2_uv_match = kp2_uv_proj.take_along_dim(row[..., None, None], -2).squeeze(-2)

    # Reconstruct the index in kp1 to match the sequence of kp2
    #  - ``idx_kp1`` is the position of the closest keypoint for kp2_uv_projected
    #  - clipping due to bad keypoints that didn't get projected onto the other image
    idx_kp1 = torch.nan_to_num(kp2_uv_match / convolution_size).int()
    idx_kp1_rel = torch.stack(_unravel_index_2d(col, (WS, WS)), 2) - 1
    idx_kp1 = idx_kp1 + idx_kp1_rel
    idx_kp1[..., 0] = idx_kp1[..., 0].clip(0, H - 1)
    idx_kp1[..., 1] = idx_kp1[..., 1].clip(0, W - 1)

    # Select kp1 matches (same order as kp2)
    PAD = (WS - 1) // 2
    kp1_uv_flat = kp1_uv_strided[..., PAD, PAD].flatten(0, 1)
    kp1_uv_match = kp1_uv_flat[tuple(idx_kp1.flatten(0, 1).T)].unflatten(0, (B, H__W))

    return flat_dists_min, idx_kp1, (kp1_uv_match, kp2_uv_match)


def _batch_point2linesegment_match(
    kp1_uv_strided: torch.Tensor, kp2_uv_proj: torch.Tensor, convolution_size: int
) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """
    Using Point-to-LineSegment distance metric, finds
    1) min distances
    2) indices of the closest keypoints for kp2_uv_projected
    3) kp1-kp2 keypoint matches (non-filtered)

    """
    # Check input
    B, H, W, C, WS, WS = kp1_uv_strided.shape
    B, H__W, D, C = kp2_uv_proj.shape
    assert H__W == H * W
    device = kp1_uv_strided.device
    assert kp1_uv_strided.device == kp2_uv_proj.device

    # Mask out points that don't project onto the other image
    kp2_mask = torch.isfinite(kp2_uv_proj).any(-1).any(-1)  # (B,H*W)
    # Rounding down gives the index of the other images point
    idx = torch.nan_to_num(kp2_uv_proj[kp2_mask] / convolution_size).int()  # (-1,D,2)
    idx[..., 0] = idx[..., 0].clip(0, H - 1)
    idx[..., 1] = idx[..., 1].clip(0, W - 1)

    # Calculate the distances
    line_points = kp2_uv_proj[kp2_mask]  # (N,D,2)
    batch_idx = (kp2_mask.int() * torch.arange(B, device=device).unsqueeze(1))[kp2_mask]
    points = (  # (N,D,2,WS*WS)
        # fmt: off
        kp1_uv_strided.movedim(0, 2)[tuple(idx.flatten(0, 1).T)]  # (N*D,B,2,WS,WS)
        .unflatten(0, idx.shape[:2])  # (N*D,...) -> (N,D,...)
        .movedim(2, 0)  # (N,D,B,...) -> (B,N,D,...)
        .take_along_dim(batch_idx[None, :, None, None, None, None], 0)  # (B,...) -> (1,...)
        .squeeze(0)  # (1,...) -> (...)
        .flatten(-2, -1)  # (...,WS,WS) -> (...,WS*WS)
    )
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
    flat_dists = distances.flatten(2)
    flat_dists_min, flat_dists_argmin = flat_dists.min(2)

    # Unravel to get index for the ``distances`` variable
    #  - ``line`` and ``line + 1``: represent the index of the two projected points kp2
    #  - ``row`` and ``col``: are required to reconstruct the index in kp1
    line, row, col = _unravel_index_3d(flat_dists_argmin, distances.shape[-3:])

    # Reconstruct the index in kp1 to match the sequence of kp2
    #  - ``idx_kp1`` is the position of the closest keypoint for kp2_uv_projected
    #  - clipping due to bad keypoints that didn't get projected onto the other image
    kp2_uv_to_round = kp2_uv_proj.take_along_dim(row[..., None, None], -2).squeeze(-2)
    idx_kp1 = torch.nan_to_num(kp2_uv_to_round / convolution_size).int()
    idx_kp1_rel = torch.stack(_unravel_index_2d(col, (WS, WS)), 2) - 1
    idx_kp1 = idx_kp1 + idx_kp1_rel
    idx_kp1[..., 0] = idx_kp1[..., 0].clip(0, H - 1)
    idx_kp1[..., 1] = idx_kp1[..., 1].clip(0, W - 1)

    # Select kp1 matches (same order as kp2)
    PAD = (WS - 1) // 2
    kp1_uv_flat = kp1_uv_strided[..., PAD, PAD].flatten(0, 1)
    kp1_uv_match = kp1_uv_flat[tuple(idx_kp1.flatten(0, 1).T)].unflatten(0, (B, H__W))

    # Select kp2 line points and calculate the match on the line segment
    kp2_uv_p1 = kp2_uv_proj.take_along_dim(line[..., None, None], -2).squeeze(-2)
    kp2_uv_p2 = kp2_uv_proj.take_along_dim(line[..., None, None] + 1, -2).squeeze(-2)
    kp2_uv_match = _project_point_on_linesegment(kp2_uv_p1, kp2_uv_p2, kp1_uv_match)

    return flat_dists_min, idx_kp1, (kp1_uv_match, kp2_uv_match)


def match_keypoints_2d(
    kp1_uv: torch.Tensor,
    kp2_uv_proj: torch.Tensor,
    convolution_size: int,
    distance_threshold: float = 0.5,
    window_size: int = 3,
    distance: Literal["p2p", "p2l"] = "p2l",
) -> tuple[torch.Tensor, ...]:
    """
    Matches keypoints ``kp1_uv`` with projected keypoints ``kp2_uv_proj``.

    Notes
    -----
    It is assumed that ``kp1_uv`` is ordered according to a meshgrid!
    Otherwise, the concept of rounding (converting to int) a keypoint in ``kp2_uv``
    may not represent the index of a close neighbour in ``kp1_uv``.

    Parameters
    ----------
    kp1_uv : torch.Tensor
        Original keypoints (see Notes!). Shape: (C, H, W)
    kp2_uv_proj : torch.Tensor
        Projected keypoints. Shape: (D, C, H, W)
    convolution_size : int
        Size of the convolution. It is used to find the keypoint indices (see Notes!).
    distance_threshold : float, optional
        Relative threshold for keypoints to be considered a "match", by default 0.5
    window_size : int, optional
        Window size for padding around neighboring kp1-keypoints, by default 3
    distance : Literal["p2p", "p2l"], optional
        Distance function to use, by default "p2l"
        If "p2p", uses Point-to-Point distance metric (less precise, slightly faster).
        If "p2l", uses Point-to-LineSegment distance metric (more precise, default).

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Matching keypoint tuple from kp1 and kp2.
    """
    return batch_match_keypoints_2d(
        kp1_uv.squeeze(0),
        kp2_uv_proj.squeeze(0),
        convolution_size,
        distance_threshold,
        window_size,
        distance,
    )[0]


def batch_match_keypoints_2d(
    kp1_uv: torch.Tensor,
    kp2_uv_proj: torch.Tensor,
    convolution_size: int,
    distance_threshold: float = 0.5,
    window_size: int = 3,
    distance: Literal["p2p", "p2l"] = "p2l",
) -> list[tuple[torch.Tensor, ...]]:
    """
    Matches keypoints ``kp1_uv`` with projected keypoints ``kp2_uv_proj``, either by
    Point-to-LineSegment (default) or Point-to-Point distance.

    Notes
    -----
    It is assumed that ``kp1_uv`` is ordered according to a meshgrid!
    Otherwise, the concept of rounding (converting to int) a keypoint in ``kp2_uv``
    may not represent the index of a close neighbour in ``kp1_uv``.

    Parameters
    ----------
    kp1_uv : torch.Tensor
        Original keypoints (see Notes!). Shape: (B, C, H, W)
    kp2_uv_proj : torch.Tensor
        Projected keypoints. Shape: (B, D, C, H, W)
    convolution_size : int
        Size of the convolution. It is used to find the keypoint indices (see Notes!).
    distance_threshold : float, optional
        Relative threshold for keypoints to be considered a "match", by default 0.5
    window_size : int, optional
        Window size for padding around neighboring kp1-keypoints, by default 3
    distance : Literal["p2p", "p2l"], optional
        Distance function to use, by default "p2l"
        If "p2p", uses Point-to-Point distance metric (less precise, slightly faster).
        If "p2l", uses Point-to-LineSegment distance metric (more precise, default).

    Returns
    -------
    list[tuple[torch.Tensor, torch.Tensor]]
        For every batch a (Tensor, Tensor) matching keypoint tuple from kp1 and kp2.

    """
    # Check input
    B, C, H, W = kp1_uv.shape
    B, D, C, H, W = kp2_uv_proj.shape
    assert C == 2
    device = kp1_uv.device
    assert kp1_uv.device == kp2_uv_proj.device
    # Permute dimensions for nicer data handling
    kp1_uv = kp1_uv.permute(0, 2, 3, 1)  # (B,H,W,2)
    kp2_uv_proj = kp2_uv_proj.view(B, D, C, H * W).permute(0, 3, 1, 2)  # (B,H*W,D,2)

    # Set padding and window size
    assert window_size % 2 == 1
    WS = window_size  # set alias
    PAD = (WS - 1) // 2

    # Create a sliding window view such that we can compare each projected keypoint
    # with the (by rounding) closest keypoint and its neighbors in the other image
    kp1_uv_padded = F.pad(kp1_uv, (0, 0, PAD, PAD, PAD, PAD), value=torch.nan)
    # -> (H+2*PAD, W+2*PAD, 2)
    kp1_uv_strided = kp1_uv_padded.unfold(1, WS, 1).unfold(2, WS, 1)
    # -> (H, W, 2, WS, WS)

    if distance == "p2p":
        min_distances, idx_kp1, kp_matches = _batch_point2point_match(
            kp1_uv_strided, kp2_uv_proj, convolution_size
        )
    elif distance == "p2l":
        min_distances, idx_kp1, kp_matches = _batch_point2linesegment_match(
            kp1_uv_strided, kp2_uv_proj, convolution_size
        )
    else:
        raise ValueError(f"Invalid {distance=}")

    # Find the closest keypoints per given index
    # while ignoring distances greater than threshold
    mask_threshold = min_distances < convolution_size * distance_threshold
    mask_closest = torch.zeros_like(mask_threshold)
    batch_idx = torch.arange(B, device=device)[:, None].broadcast_to(idx_kp1.shape[:2])[
        ..., None
    ]
    batch_indexed_idx_kp1 = torch.cat([batch_idx, idx_kp1], -1)
    mask_closest[mask_threshold] = _groupwise_smallest_values_mask(
        batch_indexed_idx_kp1[mask_threshold], min_distances[mask_threshold]
    )

    return [tuple(match[b, mask_closest[b]] for match in kp_matches) for b in range(B)]
