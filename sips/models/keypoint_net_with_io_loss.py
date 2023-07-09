# Copyright 2020 Toyota Research Institute.  All rights reserved.

from typing import Callable, Literal

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import wandb
from pytorch_lightning.loggers.wandb import WandbLogger
from torch import optim
from torchvision.transforms import ToPILImage
from typing_extensions import Self

from sips.configs.base_config import _ModelConfig
from sips.data import SonarBatch
from sips.evaluation import evaluate_keypoint_net
from sips.networks import InlierNet, KeypointNet, KeypointResnet
from sips.utils.image import normalize_2d_coordinate, normalize_sonar
from sips.utils.keypoint_matching import PSEUDO_INF, match_keypoints_2d_batch
from sips.utils.plotting import (
    COLOR_HIGHLIGHT,
    COLOR_SONAR_1,
    COLOR_SONAR_2,
    fig2img,
    plot_arcs_2d,
    plot_arcs_3d,
)
from sips.utils.point_projection import (
    uv_to_xyz_batch,
    warp_image_batch,
    xyz_to_uv_batch,
)


@torch.no_grad()
def init_weights(m: torch.nn.Module) -> None:
    """
    Initialize weights.

    Inspired by: https://adityassrana.github.io/blog/theory/2020/08/26/Weight-Init.html#Weight-Initialization:-Residual-Networks

    """
    if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
        torch.nn.init.kaiming_normal_(
            m.weight, mode="fan_out", nonlinearity="leaky_relu"
        )
    elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.GroupNorm)):
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)


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

    logger: WandbLogger | None  # type hint

    @classmethod
    def from_config(cls, config: _ModelConfig) -> Self:
        return cls(
            keypoint_loss_weight=config.keypoint_loss_weight,
            descriptor_loss_weight=config.descriptor_loss_weight,
            score_loss_weight=config.score_loss_weight,
            with_io=config.with_io,
            do_upsample=config.do_upsample,
            do_cross=config.do_cross,
            descriptor_loss=config.descriptor_loss,
            with_drop=True,
            keypoint_net_type=config.keypoint_net_type,
            opt_learn_rate=config.opt_learn_rate,
            opt_weight_decay=config.opt_weight_decay,
            sched_decay_rate=config.sched_decay_rate,
            epsilon_uv=config.epsilon_uv,
        )

    def __init__(
        self,
        keypoint_loss_weight: float = 1.0,
        descriptor_loss_weight: float = 2.0,
        score_loss_weight: float = 1.0,
        with_io: bool = True,
        do_upsample: bool = True,
        do_cross: bool = True,
        descriptor_loss: bool = True,
        with_drop: bool = True,
        keypoint_net_type: str = "KeypointNet",
        opt_learn_rate: float = 0.001,
        opt_weight_decay: float = 0.0,
        sched_decay_rate: float = 0.5,
        epsilon_uv: float = 0.5,
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not call 'self.save_hyperparameters()' since ...
        # ... 1. it's logged in 'utils/logging.py'
        # ... 2. it's may cause problems for the WandB logger

        self.keypoint_loss_weight = keypoint_loss_weight
        self.descriptor_loss_weight = descriptor_loss_weight
        self.score_loss_weight = score_loss_weight

        self.cell = 8  # Size of each output cell. Keep this fixed.
        self.border_remove = 4  # Remove points this close to the border.
        self.top_k2 = 300
        self.n_elevations = 5
        self.epsilon_uv = epsilon_uv  # threshold
        self.relax_field = int(self.epsilon_uv * self.cell)
        self.distance_metric: Literal["p2p", "p2l"] = "p2l"
        self.descriptor_loss = descriptor_loss

        # Set optimizer and scheduler parameters
        self.opt_learn_rate = opt_learn_rate
        self.opt_weight_decay = opt_weight_decay
        self.sched_decay_rate = sched_decay_rate

        # Initialize KeypointNet
        self.keypoint_net: KeypointNet | KeypointResnet
        if keypoint_net_type == "KeypointNet":
            self.use_color = False
            self.keypoint_net = KeypointNet(
                use_color=self.use_color,
                do_upsample=do_upsample,
                with_drop=with_drop,
                do_cross=do_cross,
            )
        elif keypoint_net_type == "KeypointResnet":
            self.use_color = True
            self.keypoint_net = KeypointResnet(with_drop=with_drop)
        else:
            msg = f"Keypoint net type not supported {keypoint_net_type}"
            raise NotImplementedError(msg)

        # Initialize IO-Net
        self.with_io = with_io
        self.io_net = InlierNet(blocks=4) if self.with_io else None

        # Initialize weights
        if self.io_net is not None:
            self.io_net.apply(init_weights)
        if not isinstance(self.keypoint_net, KeypointResnet):
            # KeypointResnet is skipped since we use pretrained weights!
            self.keypoint_net.apply(init_weights)

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.opt_learn_rate,
            weight_decay=self.opt_weight_decay,
        )
        lr_scheduler = {
            "scheduler": optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, factor=self.sched_decay_rate
            ),
            "monitor": "val_loss",
        }
        return [optimizer], [lr_scheduler]

    # --------------------------------------------------------------------------
    # Prediction

    _pred_out_type = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    _forward_return_type = tuple[_pred_out_type, _pred_out_type]
    __call__: Callable[..., _forward_return_type]

    def forward(self, batch: SonarBatch) -> _forward_return_type:
        # Normalize images and (optional) 1D -> 3D
        image1 = normalize_sonar(batch.image1.clone())
        image2 = normalize_sonar(batch.image2.clone())
        if self.use_color:
            image1 = image1.repeat_interleave(3, dim=1)
            image2 = image2.repeat_interleave(3, dim=1)

        # Get network outputs
        target = score_1, coord_1, desc_1 = self.keypoint_net(image1)
        source = score_2, coord_2, desc_2 = self.keypoint_net(image2)

        return target, source

    # --------------------------------------------------------------------------
    # Loss

    def _get_loss_recall(
        self,
        batch: SonarBatch,
        target_out: _pred_out_type,
        source_out: _pred_out_type,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convenience function since train/valid/test steps are similar.

        """
        # Get network outputs
        target_score, target_uv_pred, target_feat = target_out
        source_score, source_uv_pred, source_feat = source_out

        # Check dimensions
        B, _, H, W = batch.image1.shape
        _, _, HC, WC = target_score.shape

        # Initialize loss and recall (accuracy)
        loss_2d = torch.tensor(0.0, device=self.device)  # type: ignore[arg-type]

        # Normalize to uv coordinates
        target_uv_norm = normalize_2d_coordinate(target_uv_pred.clone(), H, W)
        source_uv_norm = normalize_2d_coordinate(source_uv_pred.clone(), H, W)
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
            self.cell,
            distance_threshold=torch.inf,
            distance=self.distance_metric,
            allow_multi_match=True,  # TODO: Is this the right approach?
        )
        # Apply border mask to the keypoint mask
        border_mask = torch.ones(B, HC, WC).to(mask)
        border_mask[:, : self.border_remove] = 0
        border_mask[:, -self.border_remove :] = 0
        border_mask[:, :, : self.border_remove] = 0
        border_mask[:, :, -self.border_remove :] = 0
        mask = mask & border_mask.view(mask.shape)
        # Discard keypoints for which the distance is larger than threshold
        keypoint_mask = min_distance < self.epsilon_uv * self.cell
        loc_loss = min_distance[mask & keypoint_mask].mean()
        loss_2d += self.keypoint_loss_weight * loc_loss

        source_uv_warp = source_uv_warp.permute(0, 2, 1).view(B, 2, HC, WC)
        source_uv_warp_norm = normalize_2d_coordinate(source_uv_warp.clone(), H, W)

        # 2) Descriptor head loss (per pixel level triplet loss)
        if self.descriptor_loss:
            metric_loss, recall_2d = build_descriptor_loss(
                # fmt: off
                source_feat, target_feat,
                source_uv_norm.detach(), source_uv_warp_norm.detach(), source_uv_warp,
                keypoint_mask=border_mask.view(B, HC, WC), relax_field=self.relax_field,
            )
            loss_2d += self.descriptor_loss_weight * metric_loss * 2
        else:
            _, recall_2d = build_descriptor_loss(
                # fmt: off
                source_feat, target_feat,
                source_uv_norm, source_uv_warp_norm, source_uv_warp,
                keypoint_mask=border_mask.view(B, HC, WC), relax_field=self.relax_field,
                eval_only=True,
            )

        # 3) Score head loss
        # NOTE: The code is following mostly the original implementation of KP2D
        #  (KeypointNetwithIOLoss.py) but the loss is slightly different as in the paper
        target_score_associated = target_score.squeeze(1)[
            tuple(amin_distance[mask & keypoint_mask].T)
        ]
        usp_loss = 0.5 * (
            target_score_associated + source_score.flatten(1)[mask & keypoint_mask]
        )
        usp_loss *= (
            min_distance[mask & keypoint_mask]
            - min_distance[mask & keypoint_mask].mean()
        )

        # Fill nan values with dummy values such that the gradient does not get nan.
        # Note that the nan-filled values anyway get masked out
        target_score_resampled = F.grid_sample(
            target_score,
            source_uv_warp_norm.detach().nan_to_num(),
            mode="bilinear",
            align_corners=True,
        ).view(B, HC * WC)
        mse_loss = F.mse_loss(
            target_score_resampled[mask], source_score.view(B, HC * WC)[mask]
        )

        loss_2d += self.score_loss_weight * (usp_loss.mean() + 2 * mse_loss)

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
            ).squeeze(2)
            target_feat_topk = F.grid_sample(
                target_feat, target_uv_norm_topk.unsqueeze(1), align_corners=True
            ).squeeze(2)

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

            inlier_pred = self.io_net(point_pair.permute(0, 2, 1).unsqueeze(3)).view(
                B, self.top_k2
            )

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
            inlier_gt = 2 * inlier_mask.float() - 1  # in [-1, 1], i.e. sign function

            if inlier_mask.sum() > 10:
                io_loss = 0.5 * F.mse_loss(inlier_pred, inlier_gt)
                loss_2d += io_loss

        # Sanity check
        if torch.isnan(loss_2d):
            raise RuntimeError("Loss is NaN.")

        return loss_2d, recall_2d

    # --------------------------------------------------------------------------
    # Train, test and validation

    def training_step(self, batch: SonarBatch, batch_idx: int) -> torch.Tensor:
        target_out, source_out = self(batch)
        loss, recall = self._get_loss_recall(batch, target_out, source_out)

        self.log("train_loss", loss, batch_size=batch.batch_size, sync_dist=True)
        self.log("train_recall", recall, batch_size=batch.batch_size, sync_dist=True)

        return loss

    def validation_step(self, batch: SonarBatch, batch_idx: int) -> None:
        B, _, H, W = batch.image1.shape
        (score_1, coord_1, _), (score_2, coord_2, _) = self._shared_eval(
            batch, batch_idx, "val"
        )

        # Don't make plots when there is no WandB Logger
        if not isinstance(self.logger, WandbLogger):
            return
        # Don't plot more than 64 images
        # Otherwise, there are way too many images stored in RAM, killing the terminal.
        if batch_idx >= 64 / B:
            return

        # NOTE: The image warping and keypoint matching technically allready have been
        #  done at the forward call of this class.  Therefore, it's quite redundant and
        #  could be optimized accordingly.

        # Warp coordinates to image 1
        ELEV = 5  # number of elevations
        coord_1_xyz = uv_to_xyz_batch(coord_1, batch.params1, batch.pose1, (H, W), ELEV)
        coord_2_xyz = uv_to_xyz_batch(coord_2, batch.params2, batch.pose2, (H, W), ELEV)
        coord_2_warp = xyz_to_uv_batch(coord_2_xyz, batch.params1, batch.pose1, (H, W))

        # Match keypoints
        matches_1, matches_2, dists, _, mask = match_keypoints_2d_batch(
            coord_1,
            coord_2_warp,
            self.cell,
            self.epsilon_uv,
            allow_multi_match=True,
            distance=self.distance_metric,
        )

        for b in range(B):
            to_pil = ToPILImage()
            # Find top K most confident predictions
            topk1 = score_1[b].flatten(1).topk(self.top_k2, dim=1).indices
            topk2 = score_2[b].flatten(1).topk(self.top_k2, dim=1).indices
            img_topk1 = np.array(to_pil(batch.image1[b]).convert("RGB"))
            for h, w in coord_1[b].flatten(1).take_along_dim(topk1, -1).cpu().numpy().T:
                cv2.circle(img_topk1, (round(w), round(h)), 2, (255, 0, 0), -1)
            img_topk2 = np.array(to_pil(batch.image2[b]).convert("RGB"))
            for h, w in coord_2[b].flatten(1).take_along_dim(topk2, -1).cpu().numpy().T:
                cv2.circle(img_topk2, (round(w), round(h)), 2, (255, 0, 0), -1)

            # Get 2D plot
            ax2d = plot_arcs_2d(
                [coord_1[b].flatten(1).t(), coord_2_warp[b].flatten(2).mT],
                (W, H),
                self.cell,
                [COLOR_SONAR_1, COLOR_SONAR_2],
                (matches_1[b][mask[b]].cpu(), matches_2[b][mask[b]]),
                COLOR_HIGHLIGHT,
            )
            ax2d.figure.set_size_inches(15, 15)
            img2d = fig2img(ax2d.figure, dpi=90, bbox_inches="tight", pad_inches=0)
            plt.close()

            # Reshape 3D coordinates for plotting
            xyz_1_b = coord_1_xyz[b].reshape(ELEV, 3, -1).movedim(2, 0)
            xyz_2_b = coord_2_xyz[b].reshape(ELEV, 3, -1).movedim(2, 0)
            # Get 3D overlap
            nan_mask = coord_2_warp[b].flatten(2).isnan().any(1).t()
            xyz_2_b_filtered = xyz_2_b.clone()
            xyz_2_b_filtered[nan_mask] = torch.nan
            # Get 3D plot
            ax3d = plot_arcs_3d(
                [xyz_1_b[::12], xyz_2_b[::12], xyz_2_b_filtered[::12]],
                [batch.pose1[b].position, batch.pose2[b].position, None],
                colors=[COLOR_SONAR_1, COLOR_SONAR_2, COLOR_HIGHLIGHT],
            )
            ax3d.view_init(elev=30, azim=-120, roll=0)  # type: ignore
            img3d = fig2img(ax3d.figure, dpi=200, bbox_inches="tight", pad_inches=0)
            plt.close()

            # Log images
            dists_b = dists[b, mask[b]]
            self.table.add_data(
                self.logger.experiment.step,
                wandb.Image(img_topk1),
                wandb.Image(img_topk2),
                wandb.Image(img2d),
                wandb.Image(img3d),
                dists_b.mean(),
                dists_b.size(0),
            )

    def on_validation_start(self) -> None:
        if self.logger is None:
            return

        table_columns = [
            # fmt: off
            "step", "img1", "img2", "plot2d", "plot3d",
            "mean_dist", "num_projections",
        ]
        self.table = wandb.Table(columns=table_columns)

    def on_validation_end(self) -> None:
        if self.logger is None:
            return

        step = self.logger.experiment.step
        commit = self.current_epoch % 1 == 0
        # `commit=False` just updates the last table
        self.logger.experiment.log(
            {"predictions": self.table}, step=step, commit=commit
        )

        del self.table

    def test_step(self, batch: SonarBatch, batch_idx: int) -> None:
        self._shared_eval(batch, batch_idx, "test")

    def _shared_eval(
        self, batch: SonarBatch, batch_idx: int, prefix: str
    ) -> _forward_return_type:
        assert prefix in ("val", "test")

        target_out, source_out = self(batch)
        loss, recall = self._get_loss_recall(batch, target_out, source_out)
        rep, loc, c1, c3, c5, mscore = evaluate_keypoint_net(
            batch, target_out, source_out, self.keypoint_net.cell, self.top_k2
        )
        # fmt: off
        # NOTE: If scores are renamed, must also rename in 'callbacks.py'!
        self.log(f"{prefix}_loss",               loss,   batch_size=batch.batch_size, sync_dist=True)
        self.log(f"{prefix}_recall",             recall, batch_size=batch.batch_size, sync_dist=True)
        self.log(f"{prefix}_repeatability",      rep,    batch_size=batch.batch_size, sync_dist=True)
        self.log(f"{prefix}_localization_error", loc,    batch_size=batch.batch_size, sync_dist=True)
        self.log(f"{prefix}_correctness_d1",     c1,     batch_size=batch.batch_size, sync_dist=True)
        self.log(f"{prefix}_correctness_d3",     c3,     batch_size=batch.batch_size, sync_dist=True)
        self.log(f"{prefix}_correctness_d5",     c5,     batch_size=batch.batch_size, sync_dist=True)
        self.log(f"{prefix}_matching_score",     mscore, batch_size=batch.batch_size, sync_dist=True)
        # fmt: on

        return target_out, source_out
