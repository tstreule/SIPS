import torch

from sips.data import SonarBatch
from sips.evaluation.descriptor_evaluation import (
    compute_matching_score_batch,
    dummy_compute_correctness_batch,
)
from sips.evaluation.detector_evaluation import compute_repeatability_batch


def evaluate_keypoint_net(
    batch: SonarBatch,
    target_out: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    source_out: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    cell: int,
    top_k=300,
) -> tuple[float, float, float, float, float, float]:
    # Define constants
    conf_threshold = 0.0

    # Clone the network outputs since we write in them
    target_out = tuple(tensor.clone() for tensor in target_out)  # type: ignore[assignment]
    source_out = tuple(tensor.clone() for tensor in source_out)  # type: ignore[assignment]
    # Get network outputs
    score_1, coord_1, desc_1 = target_out
    score_2, coord_2, desc_2 = source_out
    B, C, HC, WC = desc_1.shape
    B, D, H, W = batch.image1.shape

    # "Filter" (fill with nan) based on confidence threshold
    mask_1 = ~(score_1 > conf_threshold)  # don't use <= due to nan values
    mask_2 = ~(score_2 > conf_threshold)
    score_1.masked_fill_(mask_1, torch.nan)
    score_2.masked_fill_(mask_2, torch.nan)
    coord_1.masked_fill_(mask_1, torch.nan)
    coord_2.masked_fill_(mask_2, torch.nan)
    desc_1.masked_fill_(mask_1, torch.nan)
    desc_2.masked_fill_(mask_2, torch.nan)

    # Compute metrics
    _, _, repeatability_scores, localization_errors = compute_repeatability_batch(
        target_out, source_out, batch, cell, keep_top_k=top_k, matching_threshold=3
    )
    c1s, c3s, c5s = dummy_compute_correctness_batch()
    matching_scores = compute_matching_score_batch(
        target_out, source_out, batch, keep_top_k=top_k, matching_threshold=3
    )

    # Compute the batch mean and return
    return (
        sum(repeatability_scores) / B,
        sum(localization_errors) / B,
        sum(c1s) / B,
        sum(c3s) / B,
        sum(c5s) / B,
        sum(matching_scores) / B,
    )
