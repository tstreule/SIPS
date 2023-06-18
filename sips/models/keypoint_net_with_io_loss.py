# Copyright 2020 Toyota Research Institute.  All rights reserved.

from typing import Literal

import numpy as np
import numpy.typing as npt
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import optim

from sips.data import SonarBatch
from sips.evaluation import evaluate_keypoint_net
from sips.networks import InlierNet, KeypointNet, KeypointResnet
from sips.utils.image import normalize_2d_coordinate, to_color_normalized
from sips.utils.keypoint_matching import PSEUDO_INF, match_keypoints_2d_batch
from sips.utils.point_projection import warp_image_batch


def build_descriptor_loss(
    source_des: torch.Tensor,
    target_des: torch.Tensor,
    source_points: torch.Tensor,
    tar_points: torch.Tensor,
    tar_points_un: torch.Tensor,
    keypoint_mask: torch.Tensor | None = None,
    relax_field: int = 8,
    eval_only: bool = False,
):
    # Credits: https://arxiv.org/pdf/1902.11046.pdf and KP2D
    """
    Descriptor Head Loss, per-pixel level triplet loss.

    Parameters
    ----------
    source_des : torch.Tensor (B,256,H/8,W/8)
        Source image descriptors.
    target_des : torch.Tensor (B,256,H/8,W/8)
        Target image descriptors.
    source_points : torch.Tensor (B,H/8,W/8,2)
        Source image keypoints.
    tar_points : torch.Tensor (B,H/8,W/8,2)
        Target image keypoints.
    tar_points_un : torch.Tensor (B,2,H/8,W/8)
        Target image keypoints unnormalized.
    keypoint_mask : torch.Tensor (B,H/8,W/8)
        Keypoint mask.
    relax_field : int
        Compute correct matches, allowing for a few pixels tolerance (i.e. relax_field).
    eval_only : bool
        Computes only recall without the loss.

    Returns
    -------
    loss : torch.Tensor
        Descriptor loss.
    recall : torch.Tensor
        Descriptor match recall.

    """
    B, C, _, _ = source_des.shape
    device = source_des.device
    loss = torch.tensor(0.0, device=device)
    recall = torch.tensor(0.0, device=device)
    margins = 0.2

    for b in range(B):

        # Grid sample reference and target descriptor
        ref_desc = F.grid_sample(
            source_des[b].unsqueeze(0),
            source_points[b].unsqueeze(0),
            align_corners=True,
        ).squeeze()
        tar_desc = F.grid_sample(
            target_des[b].unsqueeze(0),
            # convert nan to large value such that the gradient does not get nan
            tar_points[b].unsqueeze(0).nan_to_num(PSEUDO_INF),
            align_corners=True,
        ).squeeze()
        tar_points_raw = tar_points_un[b]

        if keypoint_mask is None:
            ref_desc = ref_desc.view(C, -1)
            tar_desc = tar_desc.view(C, -1)
            tar_points_raw = tar_points_raw.view(2, -1)
        else:
            keypoint_mask_b = keypoint_mask[b].squeeze()

            n_feat = keypoint_mask_b.sum().item()
            if n_feat < 20:
                continue

            ref_desc = ref_desc[:, keypoint_mask_b]
            tar_desc = tar_desc[:, keypoint_mask_b]
            tar_points_raw = tar_points_raw[:, keypoint_mask_b]

        # Compute dense descriptor distance matrix and find nearest neighbor
        ref_desc = ref_desc.div(torch.norm(ref_desc, p=2, dim=0))
        tar_desc = tar_desc.div(torch.norm(tar_desc, p=2, dim=0))
        dmat = torch.mm(ref_desc.t(), tar_desc)
        dmat = torch.sqrt(2 - 2 * torch.clamp(dmat, min=-1, max=1))

        # Sort distance matrix
        dmat_sorted, idx = torch.sort(dmat, dim=1)

        # Compute triplet loss and recall
        candidates = idx.t()  # Candidates, sorted by descriptor distance

        # Get corresponding keypoint positions for each candidate descriptor
        match_k_x = tar_points_raw[0, candidates]
        match_k_y = tar_points_raw[1, candidates]

        # True keypoint coordinates
        true_x = tar_points_raw[0]
        true_y = tar_points_raw[1]

        # Compute recall as the number of correct matches, i.e. the first match is the correct one
        correct_matches = (match_k_x[0] - true_x).abs().eq(0) & (
            (match_k_y[0] - true_y).abs().eq(0)
        )
        recall += float(1.0 / B) * (
            float(correct_matches.float().sum()) / float(ref_desc.size(1))
        )

        if eval_only:
            continue

        # Compute correct matches, allowing for a few pixels tolerance (i.e. relax_field)
        correct_idx = (match_k_x - true_x).abs().le(relax_field) & (
            (match_k_y - true_y).abs().le(relax_field)
        )
        # Get hardest negative example as an incorrect match and with the smallest descriptor distance
        incorrect_first = dmat_sorted.t()
        incorrect_first[correct_idx] = 2.0  # largest distance is at most 2
        incorrect_first = torch.argmin(incorrect_first, dim=0)
        incorrect_first_index = candidates.gather(
            0, incorrect_first.unsqueeze(0)
        ).squeeze()

        anchor_var = ref_desc
        pos_var = tar_desc
        neg_var = tar_desc[:, incorrect_first_index]

        loss += float(1.0 / B) * F.triplet_margin_loss(
            anchor_var.t(), pos_var.t(), neg_var.t(), margin=margins
        )

    return loss, recall


class KeypointNetwithIOLoss(pl.LightningModule):
    """
    Model class encapsulating the KeypointNet and the IONet.
    """

    def __init__(
        self,
        keypoint_loss_weight: float = 1.0,
        descriptor_loss_weight: float = 2.0,
        score_loss_weight: float = 1.0,
        keypoint_net_learning_rate: float = 0.001,
        with_io: bool = True,
        use_color: bool = True,
        do_upsample: bool = True,
        do_cross: bool = True,
        descriptor_loss: bool = True,
        with_drop: bool = True,
        keypoint_net_type: str = "KeypointNet",
        opt_learn_rate: float = 0.001,
        opt_weight_decay: float = 0.0,
        sched_decay_rate: float = 0.5,
        sched_decay_frequency: int = 50,
        **kwargs,
    ) -> None:

        super().__init__()
        self.save_hyperparameters()

        self.keypoint_loss_weight = keypoint_loss_weight
        self.descriptor_loss_weight = descriptor_loss_weight
        self.score_loss_weight = score_loss_weight
        self.keypoint_net_learning_rate = keypoint_net_learning_rate

        self.cell = 8  # Size of each output cell. Keep this fixed.
        self.border_remove = 4  # Remove points this close to the border.
        self.top_k2 = 300
        self.n_elevations = 5
        self.epsilon_uv = 0.5  # threshold
        self.relax_field = int(self.epsilon_uv * self.cell)
        self.distance_metric: Literal["p2p", "p2l"] = "p2l"

        self.use_color = use_color
        self.descriptor_loss = descriptor_loss

        # Set optimizer and scheduler parameters
        self.opt_learn_rate = opt_learn_rate
        self.opt_weight_decay = opt_weight_decay
        self.sched_decay_rate = sched_decay_rate
        self.sched_decay_frequency = sched_decay_frequency

        # Initialize KeypointNet
        self.keypoint_net: KeypointNet | KeypointResnet
        if keypoint_net_type == "KeypointNet":
            self.keypoint_net = KeypointNet(
                use_color=use_color,
                do_upsample=do_upsample,
                with_drop=with_drop,
                do_cross=do_cross,
            )
        elif keypoint_net_type == "KeypointResnet":
            self.keypoint_net = KeypointResnet(with_drop=with_drop)
        else:
            msg = f"Keypoint net type not supported {keypoint_net_type}"
            raise NotImplementedError(msg)

        # Initialize IO-Net
        self.with_io = with_io
        self.io_net = InlierNet(blocks=4) if self.with_io else None

        # Other useful things to track
        self.vis: dict[str, npt.NDArray[np.float64]] = {}

    def configure_optimizers(self) -> optim.Optimizer:
        return optim.Adam(
            self.parameters(),
            lr=self.opt_learn_rate,
            weight_decay=self.opt_weight_decay,
        )

    def forward(
        self, batch: SonarBatch
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        # Normalize images and (optional) 1D -> 3D
        image1 = to_color_normalized(batch.image1.clone())
        image2 = to_color_normalized(batch.image2.clone())
        if self.keypoint_net.use_color:
            image1 = image1.repeat_interleave(3, dim=1)
            image2 = image2.repeat_interleave(3, dim=1)

        # Get network outputs
        target = score_1, coord_1, desc_1 = self.keypoint_net(image1)
        source = score_2, coord_2, desc_2 = self.keypoint_net(image2)

        return target, source

    def _get_loss_recall(
        self,
        batch: SonarBatch,
        target_out: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        source_out: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convenience function since train/valid/test steps are similar.

        """
        # Get network outputs
        target_score, target_uv_pred, target_feat = target_out
        source_score, source_uv_pred, source_feat = source_out

        B, _, H, W = batch.image1.shape
        _, _, HC, WC = target_score.shape

        # Initialize loss and recall (accuracy)
        loss_2d = torch.tensor(0.0, device=self.device)  # type: ignore[arg-type]

        # Normalize to uv coordinates
        target_uv_norm = normalize_2d_coordinate(target_uv_pred.clone(), W, H)
        source_uv_norm = normalize_2d_coordinate(source_uv_pred.clone(), W, H)
        target_uv_norm = target_uv_norm.permute(0, 2, 3, 1)
        source_uv_norm = source_uv_norm.permute(0, 2, 3, 1)
        source_uv_warp_all = warp_image_batch(
            source_uv_pred.clone(),
            batch.params2,
            batch.pose2,
            batch.params1,
            batch.pose1,
            image_resolution=(H, W),
            n_elevations=self.n_elevations,
        )

        # 1) Keypoint/Localization loss
        # Note that we select `source_uv_warp` as the closest warped coordinate points
        _, source_uv_warp, min_distance, amin_distance, mask = match_keypoints_2d_batch(
            target_uv_pred,
            source_uv_warp_all,
            self.keypoint_net.cell,
            self.epsilon_uv,
            allow_multi_match=True,
            distance=self.distance_metric,
        )
        loc_loss = min_distance[mask].mean()
        loss_2d += self.keypoint_loss_weight * loc_loss

        source_uv_warp = source_uv_warp.permute(0, 2, 1).view(B, 2, HC, WC)
        source_uv_warp_norm = normalize_2d_coordinate(source_uv_warp.clone(), W, H)
        source_uv_warp_norm = source_uv_warp_norm.permute(0, 2, 3, 1)

        # 2) Descriptor head loss (per pixel level triplet loss)
        if self.descriptor_loss:
            metric_loss, recall_2d = build_descriptor_loss(
                # fmt: off
                source_feat, target_feat,
                source_uv_norm.detach(), source_uv_warp_norm.detach(), source_uv_warp,
                keypoint_mask=mask.view(B, HC, WC), relax_field=self.relax_field,
            )
            loss_2d += self.descriptor_loss_weight * metric_loss * 2
        else:
            _, recall_2d = build_descriptor_loss(
                # fmt: off
                source_feat, target_feat,
                source_uv_norm, source_uv_warp_norm, source_uv_warp,
                keypoint_mask=mask.view(B, HC, WC), relax_field=self.relax_field,
                eval_only=True,
            )

        # 3) Score head loss
        # NOTE: The code is following mostly the original implementation of KP2D
        #  (KeypointNetwithIOLoss.py) but the loss is slightly different as in the paper
        target_score_associated = target_score.squeeze(1)[tuple(amin_distance[mask].T)]
        usp_loss = 0.5 * (target_score_associated + source_score.flatten(1)[mask])
        usp_loss *= min_distance[mask] - min_distance[mask].mean()

        # Fill nan values with dummy values such that the gradient does not get nan.
        # Note that the nan-filled values anyway get masked out
        target_score_resampled = F.grid_sample(
            target_score,
            source_uv_warp_norm.detach().nan_to_num(PSEUDO_INF),
            mode="bilinear",
            align_corners=True,
        ).view(B, HC * WC)
        mse_loss = F.mse_loss(
            target_score_resampled[mask], source_score.view(B, HC * WC)[mask]
        )

        loss_2d += self.score_loss_weight * (usp_loss.mean() + mse_loss)

        # 4) IO loss
        if self.with_io:
            assert self.io_net is not None

            # Compute IO loss
            _, top_k_indice1 = source_score.view(B, HC * WC).topk(
                self.top_k2, dim=1, largest=False
            )
            top_k_mask1 = torch.zeros(B, HC * WC).to(self.device)  # type: ignore[arg-type]
            top_k_mask1.scatter_(1, top_k_indice1, value=1)
            top_k_mask1 = top_k_mask1.gt(1e-3).view(B, HC, WC)

            _, top_k_indice2 = target_score.view(B, HC * WC).topk(
                self.top_k2, dim=1, largest=False
            )
            top_k_mask2 = torch.zeros(B, HC * WC).to(self.device)  # type: ignore[arg-type]
            top_k_mask2.scatter_(1, top_k_indice2, value=1)
            top_k_mask2 = top_k_mask2.gt(1e-3).view(B, HC, WC)

            source_uv_norm_topk = source_uv_norm[top_k_mask1].view(B, self.top_k2, 2)
            target_uv_norm_topk = target_uv_norm[top_k_mask2].view(B, self.top_k2, 2)
            source_uv_warped_norm_topk = source_uv_warp_norm[top_k_mask1].view(
                B, self.top_k2, 2
            )

            source_feat_topk = F.grid_sample(
                source_feat, source_uv_norm_topk.unsqueeze(1), align_corners=True
            ).squeeze()
            target_feat_topk = F.grid_sample(
                target_feat, target_uv_norm_topk.unsqueeze(1), align_corners=True
            ).squeeze()

            source_feat_topk = source_feat_topk.div(
                torch.norm(source_feat_topk, p=2, dim=1).unsqueeze(1)
            )
            target_feat_topk = target_feat_topk.div(
                torch.norm(target_feat_topk, p=2, dim=1).unsqueeze(1)
            )

            dmat = torch.bmm(source_feat_topk.permute(0, 2, 1), target_feat_topk)
            dmat = torch.sqrt(2 - 2 * torch.clamp(dmat, min=-1, max=1))
            # dmat_soft_min = torch.sum(dmat * dmat.mul(-1).softmax(dim=2), dim=2)
            dmat_min, dmat_min_indice = torch.min(dmat, dim=2)

            target_uv_norm_topk_associated = target_uv_norm_topk.gather(
                1, dmat_min_indice.unsqueeze(2).repeat(1, 1, 2)
            )
            point_pair = torch.cat(
                # fmt: off
                [source_uv_norm_topk, target_uv_norm_topk_associated, dmat_min.unsqueeze(2)],
                dim=2,
            )

            inlier_pred: torch.Tensor = self.io_net(
                point_pair.permute(0, 2, 1).unsqueeze(3)
            ).squeeze()

            target_uv_norm_topk_associated_raw = target_uv_norm_topk_associated.clone()
            target_uv_norm_topk_associated_raw[:, :, 0] = (
                target_uv_norm_topk_associated_raw[:, :, 0] + 1
            ) * (float(W - 1) / 2.0)
            target_uv_norm_topk_associated_raw[:, :, 1] = (
                target_uv_norm_topk_associated_raw[:, :, 1] + 1
            ) * (float(H - 1) / 2.0)

            source_uv_warped_norm_topk_raw = source_uv_warped_norm_topk.clone()
            source_uv_warped_norm_topk_raw[:, :, 0] = (
                source_uv_warped_norm_topk_raw[:, :, 0] + 1
            ) * (float(W - 1) / 2.0)
            source_uv_warped_norm_topk_raw[:, :, 1] = (
                source_uv_warped_norm_topk_raw[:, :, 1] + 1
            ) * (float(H - 1) / 2.0)

            matching_score = torch.norm(
                target_uv_norm_topk_associated_raw - source_uv_warped_norm_topk_raw,
                p=2,
                dim=2,
            )
            inlier_mask = matching_score.lt(self.relax_field)
            inlier_gt = 2 * inlier_mask.float() - 1  # in [-1, 1], i.e.  sign function

            if inlier_mask.sum() > 10:
                io_loss = 0.5 * F.mse_loss(inlier_pred, inlier_gt)
                loss_2d += io_loss

        return loss_2d, recall_2d

    def training_step(self, batch: SonarBatch, batch_idx: int):
        target_out, source_out = self(batch)
        loss, recall = self._get_loss_recall(batch, target_out, source_out)

        self.log("train_loss", loss, batch_size=batch.batch_size)
        self.log("train_recall", recall, batch_size=batch.batch_size)

        return loss

    def validation_step(self, batch: SonarBatch, batch_idx: int):
        target_out, source_out = self(batch)
        loss, recall = self._get_loss_recall(batch, target_out, source_out)
        rep, loc, c1, c3, c5, mscore = evaluate_keypoint_net(
            batch, target_out, source_out, self.keypoint_net.cell, self.top_k2
        )

        self.log("val_loss", loss, batch_size=batch.batch_size)
        self.log("val_recall", recall, batch_size=batch.batch_size)
        self.log("val_repeatability", rep, batch_size=batch.batch_size)
        self.log("val_localization_error", loc, batch_size=batch.batch_size)
        self.log("val_correctness_d1", c1, batch_size=batch.batch_size)
        self.log("val_correctness_d3", c3, batch_size=batch.batch_size)
        self.log("val_correctness_d5", c5, batch_size=batch.batch_size)
        self.log("val_mscore", mscore, batch_size=batch.batch_size)

    def test_step(self, batch: SonarBatch, batch_idx: int):
        raise NotImplementedError
