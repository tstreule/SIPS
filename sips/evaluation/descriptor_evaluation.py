from typing import Sequence

import cv2
import numpy as np
import torch

from sips.data import SonarBatch, SonarDatumPair
from sips.evaluation.detector_evaluation import _mask_top_k
from sips.utils.keypoint_matching import (
    _point2linesegment_distance,
    _point2point_distance,
)
from sips.utils.point_projection import warp_image

__all__ = [
    "dummy_compute_correctness_batch",
    "compute_matching_score_batch",
]


def _mask_inside(points: torch.Tensor, shape: Sequence[int]) -> torch.Tensor:
    """
    Keep only the set of points where at least one is inside the dimensions of shape.

    """
    # check dim
    _, _, C = points.shape
    assert C == 2

    mask = (
        (points[:, :, 0] >= 0)
        & (points[:, :, 0] < shape[0])
        & (points[:, :, 1] >= 0)
        & (points[:, :, 1] < shape[1])
    )
    return mask.any(1)


def _calc_min_distance(
    points: torch.Tensor, warped_points: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    # check dim
    _, _ = points.shape  # (N,C)
    _, _, _ = warped_points.shape  # (N,D,C)

    # prepare data
    p = points.unsqueeze(1)
    v = warped_points[:, :-1]
    w = warped_points[:, 1:]

    p2p_dists = _point2point_distance(warped_points, p, dim=dim)
    p2l_dists = _point2linesegment_distance(v, w, p, dim=dim)
    dists = torch.cat([p2p_dists, p2l_dists], dim=1)
    dists.masked_fill_(~torch.isfinite(dists), torch.inf)

    return dists.min(1).values


def dummy_compute_correctness_batch() -> tuple[list[float], list[float], list[float]]:
    # NOTE: In the case, when you have a solver for poses (similar to cv2.findHomography)
    #  you can implement this. Have a look at the function `_compute_correctness_homography`.
    return [0], [0], [0]


def compute_matching_score_batch(
    target_out: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    source_out: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    batch: SonarBatch,
    keep_top_k: int = 300,
    matching_threshold: float = 3,
) -> list[float]:
    """
    Computes the matching score.

    Compute a list of keypoints from the map, filter the list of points by keeping
    only the points that once mapped are still inside the shape of the map
    and keep at most 'keep_top_k' keypoints in the image.

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
    matching_scores : list[float]
        Matching scores.

    """
    n_elevations = 5
    B, _, H, W = batch.image1.shape

    # Get network outputs
    score_1, coord_1, desc_1 = target_out
    score_2, coord_2, desc_2 = source_out

    # "Remove" (fill with nan) that are not in keep_top_k best scores
    mask_1 = _mask_top_k(score_1, keep_top_k)
    mask_2 = _mask_top_k(score_2, keep_top_k)
    coord_1 = coord_1.masked_fill(~mask_1, torch.nan)  # we don't do it in-place
    coord_2 = coord_2.masked_fill(~mask_2, torch.nan)
    desc_1 = desc_1.masked_fill(~mask_1, torch.nan)
    desc_2 = desc_2.masked_fill(~mask_2, torch.nan)

    # Instantiate brute-force matcher
    # This part needs to be done with crossCheck=False.
    # All the matched pairs need to be evaluated without any selection.
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    # Calculate matching score for every b in batch
    matching_scores: list[float] = []
    for b in range(B):
        # Flatten the descriptors and coordinates for matching
        desc_1b = desc_1[b].flatten(1).t()
        desc_2b = desc_2[b].flatten(1).t()
        coord_1b = coord_1[b].flatten(1).t()
        coord_2b = coord_2[b].flatten(1).t()

        # Match the descriptors from 1 to 2
        matches = bf.match(desc_1b.cpu().numpy(), desc_2b.cpu().numpy())
        matches_idx1 = torch.tensor([m.queryIdx for m in matches]).long()
        matches_idx2 = torch.tensor([m.trainIdx for m in matches]).long()
        m_coord_1b = coord_1b[matches_idx1, :]
        m_coord_2b = coord_2b[matches_idx2, :]
        # Warp the keypoints and keep only those which are visible in the other image
        m_coord_1b_warp = _warp_keypoints(  # (n_points,n_elevations,2)
            m_coord_1b, batch[b], (H, W), n_elevations, invert=False
        )
        vis_mask1 = _mask_inside(m_coord_1b_warp, (H, W))  # (n_points,)
        dists1 = _calc_min_distance(m_coord_2b, m_coord_1b_warp, dim=-1)
        # Find first score
        correct1 = dists1 < matching_threshold
        count1 = torch.sum(correct1 * vis_mask1)
        score1 = count1 / torch.maximum(torch.sum(vis_mask1), torch.tensor(1.0))

        # Match the descriptors from 2 to 1
        matches = bf.match(desc_2b.cpu().numpy(), desc_1b.cpu().numpy())
        matches_idx2 = torch.tensor([m.queryIdx for m in matches]).long()
        matches_idx1 = torch.tensor([m.trainIdx for m in matches]).long()
        m_coord_1b = coord_1b[matches_idx1, :]
        m_coord_2b = coord_2b[matches_idx2, :]
        # Warp the keypoints and keep only those which are visible in the other image
        m_coord_2b_warp = _warp_keypoints(  # (n_points,n_elevations,2)
            m_coord_2b, batch[b], (H, W), n_elevations, invert=True
        )
        vis_mask2 = _mask_inside(m_coord_2b_warp, (H, W))  # (n_points,)
        dists2 = _calc_min_distance(m_coord_1b, m_coord_2b_warp, dim=-1)
        # Find first score
        correct2 = dists2 < matching_threshold
        count2 = torch.sum(correct2 * vis_mask2)
        score2 = count2 / torch.maximum(torch.sum(vis_mask2), torch.tensor(1.0))

        # Add to score
        ms = float(score1 + score2) / 2
        matching_scores.append(ms)

    return matching_scores


# --------------------------------------------------------------------------
# NOTE: Functions to be deleted as soon as `compute_correctness_batch` exists.
#  Until then, they serve as reference for future implementation and show how it can
#  be done when dealing with homographies.


def _warp_keypoints(
    points: torch.Tensor,
    sonar: "SonarDatumPair",
    shape: Sequence[int],
    n_elevations: int = 1,
    invert: bool = False,
) -> torch.Tensor:
    """
    Warp keypoints into another sonar image using just a single (!) elevation.

    Notes
    -----
    Since we are using only a single elevation, the keypoint warping may be unprecise.
    Especially, for large sonar elevation angles, this could be an issue!

    Parameters
    ----------
    points : torch.Tensor
        Keypoints to be warped. Shape: (num_points, 2)
    sonar : SonarDatumPair
        Sonar pose information for warping.
    shape : Sequence[int]
        Original image shape.
    invert : bool, optional
        If True, source and target pose are swapped, by default False

    Returns
    -------
    torch.Tensor
        Warped keypoints. Shape: (num_points, 2)

    """
    source_params = sonar.params1
    target_params = sonar.params2
    source_pose = sonar.pose1
    target_pose = sonar.pose2
    if invert:
        source_params = sonar.params2
        target_params = sonar.params1
        source_pose = sonar.pose2
        target_pose = sonar.pose1

    warped_points = (
        # fmt: off
        warp_image(
            points.t().unsqueeze(-1),
            source_params, source_pose, target_params, target_pose,
            shape, n_elevations=n_elevations,
        ).squeeze(-1).movedim(2, 0)
    )
    return warped_points


def _select_k_best(
    k: int, points: torch.Tensor, descriptors: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Select the k most probable points (and strip their probability).

    Parameters
    ----------
    k : int
        Number of keypoints to select, based on probability
    points : torch.Tensor
        Keypoint vector, consisting of (x, y, probability). Shape: (num_points, 3)
    descriptors : torch.Tensor
        Keypoint descriptors. Shape: (num_points, 256)
        Default: None

    Returns
    -------
    selected_points : torch.Tensor
        The k most probable points. Shape: (k, 2)
    selected_descriptors : torch.Tensor
        Descriptors corresponding to the k most probable keypoints. Shape: (k, 256)

    """
    argsort = points[:, 2].argsort()
    start = min(k, points.shape[0])

    sorted_prob = points[argsort, :2]
    selected_points = sorted_prob[-start:]
    sorted_desc = descriptors[argsort, :]
    selected_descriptors = sorted_desc[-start:]

    return selected_points, selected_descriptors


def _compute_correctness_homography(
    target_out: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    source_out: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    batch: SonarBatch,
    keep_top_k: int = 1000,
) -> tuple[float, float, float]:
    """
    Use the homography between two sets of keypoints and descriptors to compute
    the correctness metrics (1, 3, 5).

    """
    # Filter out predictions
    # NOTE: Since it's a dummy version, we extract one b in batch
    b = 0
    sonar = batch[b]
    shape = sonar.image1.shape
    score_1, coord_1, desc_1 = [out[b] for out in target_out]
    score_2, coord_2, desc_2 = [out[b] for out in source_out]

    C, _, _ = desc_1.shape
    keypoints1 = torch.cat([coord_1, score_1], dim=1).view(3, -1).t()
    keypoints2 = torch.cat([coord_2, score_2], dim=1).view(3, -1).t()
    descriptors1 = desc_1.view(C, -1).t()
    descriptors2 = desc_2.view(C, -1).t()

    # Warp the keypoints and keep only those which are visible in the other image
    # NOTE: We set `n_elevations=1` for simplicity.
    #  Hence the distance between probably is not very accurate.
    keypoints1_warp = _warp_keypoints(keypoints1[:, :2], sonar, shape)
    keypoints2_warp = _warp_keypoints(keypoints2[:, :2], sonar, shape, invert=True)
    mask1 = _mask_inside(keypoints1_warp.unsqueeze(1), shape)
    mask2 = _mask_inside(keypoints2_warp.unsqueeze(1), shape)
    # Keep only the points shared between the two views
    kp1_, desc1_ = _select_k_best(keep_top_k, keypoints1[mask1], descriptors1[mask1])
    kp2_, desc2_ = _select_k_best(keep_top_k, keypoints2[mask2], descriptors2[mask2])
    # Convert to numpy array
    kp1, desc1 = kp1_.cpu().numpy(), desc1_.cpu().numpy()
    kp2, desc2 = kp2_.cpu().numpy(), desc2_.cpu().numpy()

    # Brute-force match descriptors
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(desc1, desc2)
    matches_idx1 = np.array([m.queryIdx for m in matches])
    matches_idx2 = np.array([m.trainIdx for m in matches])
    m_keypoints1 = kp1[matches_idx1, :]
    m_keypoints2 = kp2[matches_idx2, :]

    if m_keypoints1.shape[0] < 4 or m_keypoints2.shape[0] < 4:
        return 0, 0, 0

    # Estimate the homography between the matches using RANSAC
    H, _ = cv2.findHomography(m_keypoints1, m_keypoints2, cv2.RANSAC, 3, maxIters=5000)
    if H is None:
        return 0, 0, 0

    # Compute correctness
    corners = np.array(
        # fmt: off
        [[0,            0,            1],
         [0,            shape[1] - 1, 1],
         [shape[0] - 1, 0,            1],
         [shape[0] - 1, shape[1] - 1, 1]]
    )
    # NOTE: We don't have a real homography at hand.
    real_warped_corners = np.dot(corners, np.transpose(real_H))  # type: ignore[name-defined]
    real_warped_corners = real_warped_corners[:, :2] / real_warped_corners[:, 2:]
    warped_corners = np.dot(corners, np.transpose(H))
    warped_corners = warped_corners[:, :2] / warped_corners[:, 2:]

    mean_dist = np.mean(np.linalg.norm(real_warped_corners - warped_corners, axis=1))
    correctness1 = float(mean_dist <= 1)
    correctness3 = float(mean_dist <= 3)
    correctness5 = float(mean_dist <= 5)

    return correctness1, correctness3, correctness5
