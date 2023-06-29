from typing import Sequence

import torch

from sips.data import SonarBatch
from sips.utils.keypoint_matching import match_keypoints_2d_batch
from sips.utils.point_projection import warp_image_batch

__all__ = ["compute_repeatability_batch"]


def _mask_inside(points: torch.Tensor, shape: Sequence[int]) -> torch.Tensor:
    """
    Keep only the set of points where at least one is inside the dimensions of shape.

    """
    # check dim
    _, _, C, _, _ = points.shape
    assert C == 2

    mask = (
        (points[:, :, 0] >= 0)
        & (points[:, :, 0] < shape[0])
        & (points[:, :, 1] >= 0)
        & (points[:, :, 1] < shape[1])
    )
    return mask.any(1, keepdim=True)


def _mask_top_k(scores: torch.Tensor, top_k: int) -> torch.Tensor:
    """
    Keep only the points whose scores are within the top_k (for each b in batch).

    """
    # check dim
    B, C, _, _ = scores.shape
    assert C == 1

    top_k_idx = scores.flatten(1).argsort()[:, -top_k:]
    mask = torch.full_like(scores, 0, dtype=torch.bool)
    mask.view(B, -1).scatter_(1, top_k_idx, True)
    return mask


def compute_repeatability_batch(
    target_out: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    source_out: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    batch: SonarBatch,
    cell: int,
    keep_top_k: int = 300,
    matching_threshold: float = 3,
) -> tuple[list[int], list[int], list[float], list[float]]:
    """
    Computes the ratio between the number of point-to-point correspondences and
    the minimum number of points detected in the images.

    References
    ----------
    - DeTone et al. (2018), "SuperPoint: Self-Supervised Interest Point Destection and
      Description", Appendix,
      [Online](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w9/DeTone_SuperPoint_Self-Supervised_Interest_CVPR_2018_paper.pdf)

    Parameters
    ----------
    target_out : tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Predicted target scores, coordinates and descriptors.
    source_out : tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Predicted source scores, coordinates and descriptors.
    batch : SonarBatch
        Sonar data batch.
    cell : int
        Cell size, i.e., convolution size.
    keep_top_k : int, optional
        Number of best points to keep for computing the score.
        Default: 300
    matching_threshold : float, optional
        Distance threshold (in pixels) for two points to be considered a match.
        Default: 3

    Returns
    -------
    N1s : list[int]
    N2s : list[int]
    repeatability_scores : list[float]
        Mean repeatability score.
    localization_errors : list[float]
        Mean localization error.

    """
    n_elevations = 5
    B, _, H, W = batch.image1.shape

    # Get network outputs
    score_1, coord_1, _ = target_out
    score_2, coord_2, _ = source_out

    # Project the images onto each other
    coord_1_proj = warp_image_batch(
        # fmt: off
        coord_1, batch.params1, batch.pose1, batch.params2, batch.pose2,
        image_resolution=(H, W), n_elevations=n_elevations,
    )
    coord_2_proj = warp_image_batch(
        # fmt: off
        coord_2, batch.params2, batch.pose2, batch.params1, batch.pose1,
        image_resolution=(H, W), n_elevations=n_elevations,
    )

    # "Remove" (fill with nan) keypoints outside
    mask_1 = _mask_inside(coord_1_proj, (H, W))
    mask_2 = _mask_inside(coord_2_proj, (H, W))
    score_1 = score_1.masked_fill(mask_1, 0.0)  # we don't do it in-place
    score_2 = score_2.masked_fill(mask_2, 0.0)

    # "Remove" (fill with nan) that are not in keep_top_k best scores
    mask_1 = _mask_top_k(score_1, keep_top_k)
    mask_2 = _mask_top_k(score_2, keep_top_k)
    coord_1 = coord_1.masked_fill(mask_1, torch.nan)  # we don't do it in-place
    coord_2 = coord_2.masked_fill(mask_2, torch.nan)

    # Compute minimum distances
    _, _, min1, _, _ = match_keypoints_2d_batch(coord_1, coord_2_proj, cell, torch.inf)
    _, _, min2, _, _ = match_keypoints_2d_batch(coord_2, coord_1_proj, cell, torch.inf)

    # Compute repeatability
    n1s: list[int] = []
    n2s: list[int] = []
    repeatability_scores: list[float] = []
    localization_errors: list[float] = []
    for b in range(B):
        min1b = min1[b, mask_1[b].flatten()]
        min2b = min2[b, mask_2[b].flatten()]
        N1 = min1b.shape[0]
        N2 = min2b.shape[0]

        correct1b = min1b <= matching_threshold
        correct2b = min2b <= matching_threshold
        count1b = torch.sum(correct1b)
        count2b = torch.sum(correct2b)

        le1b = min1b[correct1b].sum()
        le2b = min2b[correct2b].sum()

        # Repeatability measures the probability that a point is detected in the second image.
        # higher RS is better
        if N1 + N2 > 0:
            rs = float(count1b + count2b) / (N1 + N2)
        else:
            rs = 0.0
        # Localization Error is the average pixel distance between corresponding points.
        # LE is between 0 and matching_threshold, and lower LE is better
        if count1b + count2b > 0:
            loc_err = float(le1b + le2b) / float(count1b + count2b)
        else:
            loc_err = float(matching_threshold)

        # Append
        n1s.append(N1)
        n2s.append(N2)
        repeatability_scores.append(rs)
        localization_errors.append(loc_err)

    return n1s, n2s, repeatability_scores, localization_errors
