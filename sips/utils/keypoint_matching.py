from typing import Iterable, Literal

import torch
import torch.nn.functional as F

__all__ = ["match_keypoints_2d", "match_keypoints_2d_batch"]


# ==============================================================================
# Utils


PSEUDO_INF = 1e6


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
    idx_smallest_values = grouped_data[..., w + 1].gather(1, argmins).flatten().long()

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
    l2 = l2.clamp(1e-6)  # handle v == w case
    t = (torch.einsum("...i,...i->...", p - v, w - v) / l2).clamp(0, 1)  # dot product
    projection = v + t[..., None] * (w - v)

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
    # Fill nan values with dummy values such that the gradient does not get nan
    # then fill those dummy distances with nan values to obtain true distance.
    nan_mask = ~(torch.isfinite(p1).all(dim) & torch.isfinite(p2).all(dim))
    diff = p1.nan_to_num() - p2.nan_to_num()
    distance = torch.linalg.vector_norm(diff, dim=dim)
    # Clone to prevent the following error due to inplace operation:
    #     RuntimeError: one of the variables needed for gradient computation
    #     has been modified by an inplace operation [...]
    distance = distance.clone()
    distance[nan_mask] = torch.nan
    return distance


def _point2linesegment_distance(
    v: torch.Tensor, w: torch.Tensor, p: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """
    Calculates the Point-to-LineSegment distances.

    Every point ``p`` is projected onto the line segments ``v``-``w``.

    """
    # Fill nan values with dummy values such that the gradient does not get nan
    # then fill those dummy distances with nan values to obtain true distance.
    nan_mask = ~(
        torch.isfinite(v).all(dim)
        & torch.isfinite(w).all(dim)
        & torch.isfinite(p).all(dim)
    )
    projection = _project_point_on_linesegment(
        v.nan_to_num(), w.nan_to_num(), p.nan_to_num(), dim=dim
    )
    # Clone to prevent the following error due to inplace operation:
    #     RuntimeError: one of the variables needed for gradient computation
    #     has been modified by an inplace operation [...]
    projection = projection.clone()
    projection[nan_mask.unsqueeze(dim).broadcast_to(projection.shape)] = torch.nan

    return _point2point_distance(p, projection, dim=dim)


# ==============================================================================
# Keypoint matching


def _point2point_match_batch(
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
    assert C == 2 and D >= 1
    assert H__W == H * W
    device = kp1_uv_strided.device
    assert kp1_uv_strided.device == kp2_uv_proj.device

    # Mask out points that don't project onto the other image
    kp2_mask = torch.isfinite(kp2_uv_proj).any(-1)  # (B,H*W,D)
    # Rounding down gives the index of the other images point
    idx = torch.nan_to_num(kp2_uv_proj[kp2_mask] / convolution_size).long()  # (-1,2)
    idx[..., 0] = idx[..., 0].clip(0, H - 1)
    idx[..., 1] = idx[..., 1].clip(0, W - 1)

    # Calculate the distances
    batch_idx = kp2_mask.long() * torch.arange(B, device=device)[:, None, None]
    batch_idx = batch_idx[kp2_mask]
    masked_dists = _point2point_distance(
        kp2_uv_proj[kp2_mask][..., None, None],
        kp1_uv_strided.movedim(0, 2)[tuple(idx.T)]
        .movedim(1, 0)
        .take_along_dim(batch_idx[None, :, None, None, None], 0)
        .squeeze(0),
        dim=1,
    ).view(-1, WS**2)
    distances = torch.full((*kp2_mask.shape, WS**2), torch.inf, device=device)
    distances[kp2_mask] = masked_dists.nan_to_num(torch.inf)

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
    idx_kp1 = torch.nan_to_num(kp2_uv_match // convolution_size).long()
    idx_kp1 += torch.stack(_unravel_index_2d(col, (WS, WS)), 2) - 1  # relative index
    idx_kp1[..., 0] = idx_kp1[..., 0].clip(0, H - 1)
    idx_kp1[..., 1] = idx_kp1[..., 1].clip(0, W - 1)
    # Add batch column
    batch_idx = torch.arange(B, device=device)[:, None]
    batch_idx = batch_idx.broadcast_to(idx_kp1.shape[:2])[..., None]
    batch_idx_kp1 = torch.cat([batch_idx, idx_kp1], -1)

    # Select kp1 matches (same order as kp2)
    PAD = (WS - 1) // 2
    kp1_uv = kp1_uv_strided[..., PAD, PAD]
    kp1_uv_match = kp1_uv[tuple(batch_idx_kp1.flatten(0, 1).T)].unflatten(0, (B, H__W))

    return flat_dists_min, batch_idx_kp1, (kp1_uv_match, kp2_uv_match)


def _point2linesegment_match_batch(
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
    assert C == 2 and D >= 1
    device = kp1_uv_strided.device
    assert kp1_uv_strided.device == kp2_uv_proj.device

    # Mask out points that don't project onto the other image
    # or that don't make up a line (only one elevation was projected).
    # In the latter case, let ``_point2point_distance()`` handle it.
    mask_finite = torch.isfinite(kp2_uv_proj).any(-1).sum(-1)
    kp2_mask = mask_finite.gt(1)  # (B,H*W)
    kp2_mask_p2p = mask_finite.eq(1)  # (B,H*W)
    # Rounding down gives the index of the other images point
    idx = torch.nan_to_num(kp2_uv_proj[kp2_mask] / convolution_size).int()  # (-1,D,2)
    idx[..., 0] = idx[..., 0].clip(0, H - 1)
    idx[..., 1] = idx[..., 1].clip(0, W - 1)

    # Calculate the distances
    line_points = kp2_uv_proj[kp2_mask]  # (N,D,2)
    batch_idx = (kp2_mask.int() * torch.arange(B, device=device).unsqueeze(1))[kp2_mask]
    N = len(idx)
    points = (  # (N,D,2,WS*WS)
        # fmt: off
        kp1_uv_strided.movedim(0, 2)[tuple(idx.flatten(0, 1).T)]  # (N*D,B,2,WS,WS)
        .reshape(N, D, B, 2*WS*WS)
        .take_along_dim(batch_idx[:, None, None, None], 2)  # (N,D,1,2,WS,WS)
        .view(N, D, 2, WS*WS)
    )
    masked_distances = _point2linesegment_distance(  # (N,D-1,D,WS*WS)
        line_points[:, :-1, None, :, None],
        line_points[:, 1:, None, :, None],
        points[:, None],
        dim=3,
    )
    distances = torch.full(  # (B,H*W,D-1,D,WS*WS)
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
    kp2_uv_to_round = kp2_uv_proj.take_along_dim(row[..., None, None], 2).squeeze(2)
    idx_kp1 = torch.nan_to_num(kp2_uv_to_round // convolution_size).long()
    idx_kp1 += torch.stack(_unravel_index_2d(col, (WS, WS)), 2) - 1  # relative index
    idx_kp1[..., 0] = idx_kp1[..., 0].clip(0, H - 1)
    idx_kp1[..., 1] = idx_kp1[..., 1].clip(0, W - 1)
    # Add batch column
    batch_idx = torch.arange(B, device=device)[:, None]
    batch_idx = batch_idx.broadcast_to(idx_kp1.shape[:2])[..., None]
    batch_idx_kp1 = torch.cat([batch_idx, idx_kp1], -1)

    # Select kp1 matches (same order as kp2)
    PAD = (WS - 1) // 2
    kp1_uv = kp1_uv_strided[..., PAD, PAD]
    kp1_uv_match = kp1_uv[tuple(batch_idx_kp1.flatten(0, 1).T)].unflatten(0, (B, H__W))

    # Select kp2 line points and calculate the match on the line segment
    kp2_uv_p1 = kp2_uv_proj.take_along_dim(line[..., None, None], -2).squeeze(-2)
    kp2_uv_p2 = kp2_uv_proj.take_along_dim(line[..., None, None] + 1, -2).squeeze(-2)
    kp2_uv_match = _project_point_on_linesegment(kp2_uv_p1, kp2_uv_p2, kp1_uv_match)

    # Use point2point distance where only one elevation got projected
    a, b, (c, d) = _point2point_match_batch(
        kp1_uv_strided, kp2_uv_proj, convolution_size
    )
    flat_dists_min[kp2_mask_p2p] = a[kp2_mask_p2p]
    batch_idx_kp1[kp2_mask_p2p] = b[kp2_mask_p2p]
    kp1_uv_match[kp2_mask_p2p] = c[kp2_mask_p2p]
    kp2_uv_match[kp2_mask_p2p] = d[kp2_mask_p2p]

    return flat_dists_min, batch_idx_kp1, (kp1_uv_match, kp2_uv_match)


def match_keypoints_2d(
    kp1_uv: torch.Tensor,
    kp2_uv_proj: torch.Tensor,
    convolution_size: int,
    distance_threshold: float = 0.5,
    allow_multi_match: bool = True,
    window_size: int = 3,
    distance: Literal["p2p", "p2l"] = "p2l",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        Original keypoints (see Notes!). Shape: (2, H, W)
    kp2_uv_proj : torch.Tensor
        Projected keypoints. Shape: (D, 2, H, W)
    convolution_size : int
        Size of the convolution. It is used to find the keypoint indices (see Notes!).
    distance_threshold : float, optional
        Relative threshold for keypoints to be considered a "match", by default 0.5
    allow_multi_match : bool, optional
        If False, will only accept the closest match per kp1 keypoint.
        If True, multiple kp2 keypoints may match with a single kp1 keypoint.
        Default: True
    window_size : int, optional
        Window size for padding around neighboring kp1-keypoints, by default 3
    distance : Literal["p2p", "p2l"], optional
        Distance function to use, by default "p2l"
        If "p2p", uses Point-to-Point distance metric (less precise, slightly faster).
        If "p2l", uses Point-to-LineSegment distance metric (more precise, default).

    Returns
    -------
    kp1_matches : torch.Tensor
        Matching keypoints from kp1 (not yet masked). Shape: (H*W, 2)
    kp2_matches : torch.Tensor
        Matching keypoints from kp2 (not yet masked). Shape: (H*W, 2)
    min_distances : torch.Tensor
        Distances for keypoint matches (not yet masked). Shape: (H*W,)
    argmin_distances : torch.Tensor
        Indices for ``min_distances`` (not yet masked). Shape: (H*W, 2)
    mask : torch.Tensor
        Threshold and ensure-single-match mask. Shape: (H*W,)

    """
    batched_out = match_keypoints_2d_batch(
        kp1_uv=kp1_uv.unsqueeze(0),
        kp2_uv_proj=kp2_uv_proj.unsqueeze(0),
        convolution_size=convolution_size,
        distance_threshold=distance_threshold,
        allow_multi_match=allow_multi_match,
        window_size=window_size,
        distance=distance,
    )
    kp1_matches, kp2_matches, min_distances, argmin_distances, mask = batched_out
    return (
        kp1_matches.squeeze(0),
        kp2_matches.squeeze(0),
        min_distances.squeeze(0),
        argmin_distances.squeeze(0)[:, (1, 2)],  # don't need batch index
        mask.squeeze(0),
    )


def _sanity_check_kp1_uv(kp1_uv: torch.Tensor, cell: int, H: int, W: int) -> None:
    """
    Check whether the values of kp1_uv are arranged according to a meshgrid.

    """
    # Check dim
    _, C, _, _ = kp1_uv.shape
    assert C == 2

    # Use only first batch item
    uv0 = kp1_uv[0]

    # Get index
    idx = (uv0 // cell).long().flatten(1)
    idx[0] = idx[0].clip(0, H - 1)
    idx[1] = idx[1].clip(0, W - 1)

    # Reconstruct uv0 using the index
    uv0_rec = uv0.movedim(0, 2)[tuple(idx)].t().view(uv0.shape)

    # Check if allclose while ignoring NaN values
    msg = (
        "The original keypoints are not ordered according to a meshgrid. "
        "This means that the concept of rounding (converting to int) does not "
        "represent the index of itself."
    )
    mask = torch.isfinite(uv0).all(0) & torch.isfinite(uv0_rec).all(0)
    torch.testing.assert_close(
        # fmt: off
        uv0[:, mask], uv0_rec[:, mask],
        equal_nan=True, msg=msg,
        # Set a very high tolerance as the keypoints do not always land on themselves
        rtol=3 * cell, atol=3 * cell,
    )


def match_keypoints_2d_batch(
    kp1_uv: torch.Tensor,
    kp2_uv_proj: torch.Tensor,
    convolution_size: int,
    distance_threshold: float = 0.5,
    allow_multi_match: bool = True,
    window_size: int = 3,
    distance: Literal["p2p", "p2l"] = "p2l",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        Original keypoints (see Notes!). Shape: (B, 2, H, W)
    kp2_uv_proj : torch.Tensor
        Projected keypoints. Shape: (B, D, 2, H, W)
    convolution_size : int
        Size of the convolution. It is used to find the keypoint indices (see Notes!).
    distance_threshold : float, optional
        Relative threshold for keypoints to be considered a "match", by default 0.5
    allow_multi_match : bool, optional
        If False, will only accept the closest match per kp1 keypoint.
        If True, multiple kp2 keypoints may match with a single kp1 keypoint.
        Default: True
    window_size : int, optional
        Window size for padding around neighboring kp1-keypoints, by default 3
    distance : Literal["p2p", "p2l"], optional
        Distance function to use, by default "p2l"
        If "p2p", uses Point-to-Point distance metric (less precise, slightly faster).
        If "p2l", uses Point-to-LineSegment distance metric (more precise, default).

    Returns
    -------
    kp1_matches : torch.Tensor
        Matching keypoints from kp1 (not yet masked). Shape: (B, H*W, 2)
    kp2_matches : torch.Tensor
        Matching keypoints from kp2 (not yet masked). Shape: (B, H*W, 2)
    min_distances : torch.Tensor
        Distances for keypoint matches (not yet masked). Shape: (B, H*W)
    argmin_distances : torch.Tensor
        Indices for ``min_distances`` (not yet masked). Shape: (B, H*W, 3)
    mask : torch.Tensor
        Threshold and ensure-single-match mask. Shape: (B, H*W)

    """
    # Check input
    B, C, H, W = kp1_uv.shape
    B, D, C, H, W = kp2_uv_proj.shape
    assert C == 2
    assert kp1_uv.device == kp2_uv_proj.device
    try:
        # Check if 'kp1_uv' is meshgrid-like
        _sanity_check_kp1_uv(kp1_uv, convolution_size, H, W)
    except AssertionError:
        # Transpose and do the sanity check again to avoid infitite try-except loop
        kp1_uv = kp1_uv.mT.contiguous()
        kp2_uv_proj = kp2_uv_proj.mT.contiguous()
        _sanity_check_kp1_uv(kp1_uv, convolution_size, H, W)

        # Use swapped input
        output = match_keypoints_2d_batch(
            kp1_uv,
            kp2_uv_proj,
            convolution_size,
            distance_threshold,
            allow_multi_match,
            window_size,
            distance,
        )
        kp1_matches, kp2_matches, min_distances, argmin_distances, mask = output

        # Undo the swap
        kp1_matches = kp1_matches.view(B, H, W, 2).swapaxes_(1, 2).flatten(1, 2)
        kp2_matches = kp2_matches.view(B, H, W, 2).swapaxes_(1, 2).flatten(1, 2)
        min_distances = min_distances.view(B, H, W).mT.flatten(1, 2)
        argmin_distances = (
            argmin_distances.view(B, H, W, 3).swapaxes_(1, 2).flatten(1, 2)
        )
        mask = mask.view(B, H, W).mT.flatten(1, 2)

        return (kp1_matches, kp2_matches, min_distances, argmin_distances, mask)

    # Permute dimensions for nicer data handling
    kp1_uv = kp1_uv.permute(0, 2, 3, 1)  # (B,H,W,2)
    kp2_uv_proj = kp2_uv_proj.view(B, D, C, H * W).permute(0, 3, 1, 2)  # (B,H*W,D,2)

    # Set padding and window size
    assert window_size % 2 == 1
    WS = window_size  # set alias
    PAD = (WS - 1) // 2

    # Create a sliding window view such that we can compare each projected keypoint
    # with the (by rounding) closest keypoint and its neighbors in the other image
    kp1_uv_padded = F.pad(kp1_uv, (0, 0, PAD, PAD, PAD, PAD), mode="replicate")
    # -> (B, H+2*PAD, W+2*PAD, 2)
    kp1_uv_strided = kp1_uv_padded.unfold(1, WS, 1).unfold(2, WS, 1)
    # -> (B, H, W, 2, WS, WS)

    # Calculate distances
    if distance == "p2p":
        min_distances, argmin_distances, kp_matches = _point2point_match_batch(
            kp1_uv_strided, kp2_uv_proj, convolution_size
        )
    elif distance == "p2l":
        min_distances, argmin_distances, kp_matches = _point2linesegment_match_batch(
            kp1_uv_strided, kp2_uv_proj, convolution_size
        )
    else:
        raise ValueError(f"Invalid {distance=}")
    kp1_matches, kp2_matches = kp_matches

    # Ignore distances greater than threshold
    mask_epsilon = min_distances.lt(convolution_size * distance_threshold)
    # Allow multiple matches per kp1 keypoint
    if allow_multi_match:
        mask = mask_epsilon
    # Only allow the closest kp1 keypoint matches
    else:
        mask = mask_epsilon.clone()
        mask[mask_epsilon] = _groupwise_smallest_values_mask(
            argmin_distances[mask_epsilon], min_distances[mask_epsilon]
        )

    # Return mask separately since number of matches may differ among images (b in B)
    return kp1_matches, kp2_matches, min_distances, argmin_distances, mask
