import cv2 as cv
import numpy as np
import numpy.typing as npt
import torch
import tqdm
from pytorch_lightning import seed_everything
from torch import nn
from torch.utils.data import DataLoader

from sips.configs.parsing import parse_train_file
from sips.data import SonarDatumPair
from sips.datasets.data_module import SonarDataModule
from sips.evaluation.descriptor_evaluation import compute_matching_score_batch
from sips.evaluation.detector_evaluation import compute_repeatability_batch
from sips.models.keypoint_net_with_io_loss import build_descriptor_loss
from sips.utils.image import normalize_2d_coordinate
from sips.utils.keypoint_matching import _unravel_index_2d, match_keypoints_2d_batch
from sips.utils.point_projection import warp_image_batch

try:
    import typer
except ModuleNotFoundError:
    from sips.utils import typer  # type: ignore[no-redef]

PSEUDO_NEG_INF = -1e6

app = typer.Typer()


def format_scores_coords_descs(
    kps: tuple[cv.KeyPoint],
    desc: npt.NDArray[np.uint8],
    desc_size: int,
    conv_size: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Create a fake image with the highest orb response at each pixel, if no
    # key point was found at a pixel leave at PSEUDO_NEG_INF. Note that ORB finds
    # subpixel-level keypoints, by casting to int we ensure that all keypoints
    # within the same pixel correspond to that pixel.
    kp_orb1_uv = torch.full((512, 512), PSEUDO_NEG_INF).to(torch.float)
    kp_pt = torch.full((2, 512, 512), -1).to(torch.float)
    kp_desc = torch.full((desc_size, 512, 512), -1).to(torch.float)
    for idx, kp in enumerate(kps):
        # If the value at a pixel is bigger than PSEUDO_NEG_INF that means that we already
        # stored a response at that pixel. Keep the bigger response
        if kp_orb1_uv[int(kp.pt[0]), int(kp.pt[1])] > PSEUDO_NEG_INF:
            if kp_orb1_uv[int(kp.pt[0]), int(kp.pt[1])] > kp.response:
                continue
            else:
                kp_orb1_uv[int(kp.pt[0]), int(kp.pt[1])] = kp.response
                kp_pt[0, int(kp.pt[0]), int(kp.pt[1])] = kp.pt[0]
                kp_pt[1, int(kp.pt[0]), int(kp.pt[1])] = kp.pt[1]
                kp_desc[:, int(kp.pt[0]), int(kp.pt[1])] = torch.tensor(desc[idx])
        else:
            kp_orb1_uv[int(kp.pt[0]), int(kp.pt[1])] = kp.response
            kp_pt[0, int(kp.pt[0]), int(kp.pt[1])] = kp.pt[0]
            kp_pt[1, int(kp.pt[0]), int(kp.pt[1])] = kp.pt[1]
            kp_desc[:, int(kp.pt[0]), int(kp.pt[1])] = torch.tensor(desc[idx])
    maxpool = nn.MaxPool2d(conv_size, stride=conv_size, return_indices=True)
    pool, indices = maxpool(kp_orb1_uv.reshape((1, 512, 512)).to(torch.float))
    mask = pool > PSEUDO_NEG_INF
    indices_ = indices[mask]
    row, col = _unravel_index_2d(indices_, [4096, 512])
    scores = torch.full((1, 64, 64), torch.nan)
    coords = torch.full((2, 64, 64), torch.nan)
    descs = torch.full((desc_size, 64, 64), torch.nan)
    scores[mask] = kp_orb1_uv[row, col]
    coords[:, mask.squeeze()] = kp_pt[:, row, col]
    descs[:, mask.squeeze()] = kp_desc[:, row, col]
    return scores, coords, descs


def compute_recall_batch(
    target_out,
    source_out,
    batch,
    n_elevations=5,
    conv_size=8,
    border_remove=4,
    relax_field=4,
) -> float:
    target_score, target_points, target_des = target_out
    _, source_points, source_des = source_out

    # Check dimensions
    B, _, H, W = batch.image1.shape
    _, _, HC, WC = target_score.shape

    source_points_norm = normalize_2d_coordinate(source_points.clone(), H, W)

    source_points_warp_all = warp_image_batch(
        source_points.clone(),
        batch.params2,
        batch.pose2,
        batch.params1,
        batch.pose1,
        image_resolution=(H, W),
        n_elevations=n_elevations,
    )

    _, source_points_warp, min_distance, _, mask = match_keypoints_2d_batch(
        target_points,
        source_points_warp_all,
        conv_size,
        distance_threshold=torch.inf,
        distance="p2l",
        allow_multi_match=True,
    )

    border_mask = torch.ones(B, HC, WC).to(mask)
    border_mask[:, :border_remove] = 0
    border_mask[:, -border_remove:] = 0
    border_mask[:, :, :border_remove] = 0
    border_mask[:, :, -border_remove:] = 0

    source_points_warp = source_points_warp.permute(0, 2, 1).view(B, 2, HC, WC)
    source_points_warp_norm = normalize_2d_coordinate(source_points_warp.clone(), H, W)
    keypoint_mask = min_distance < 4

    _, recall = build_descriptor_loss(
        source_des,
        target_des,
        source_points_norm,
        source_points_warp_norm,
        source_points_warp,
        keypoint_mask=(mask & border_mask.view(mask.shape)).view(B, HC, WC),
        relax_field=relax_field,
        eval_only=True,
    )
    return float(recall)


def evaluate(
    model_name: str,
    dataloader: DataLoader[SonarDatumPair],
    nfeatures: int,
    matching_threshold: int,
    conv_size: int = 8,
):
    detector_eval = []
    descriptor_eval = []
    recall_eval = []

    if model_name == "ORB":
        model = cv.ORB_create(nfeatures=nfeatures)
        desc_size = 32
    elif model_name == "SIFT":
        model = cv.SIFT_create(nfeatures=nfeatures)
        desc_size = 128
    else:
        raise ValueError("Unknown model")

    for batch in tqdm.tqdm(dataloader):
        out1 = (
            torch.full((batch.batch_size, 1, 64, 64), torch.nan),
            torch.full((batch.batch_size, 2, 64, 64), torch.nan),
            torch.full((batch.batch_size, desc_size, 64, 64), torch.nan),
        )
        out2 = (
            torch.full((batch.batch_size, 1, 64, 64), torch.nan),
            torch.full((batch.batch_size, 2, 64, 64), torch.nan),
            torch.full((batch.batch_size, desc_size, 64, 64), torch.nan),
        )
        for datum_pair_idx in range(batch.batch_size):
            im1 = batch.image1[datum_pair_idx, ...]
            im2 = batch.image2[datum_pair_idx, ...]
            kp1, des1 = model.detectAndCompute(np.array(im1.squeeze()), None)
            kp2, des2 = model.detectAndCompute(np.array(im2.squeeze()), None)
            scores1, coords1, descs1 = format_scores_coords_descs(kp1, des1, desc_size)
            scores2, coords2, descs2 = format_scores_coords_descs(kp2, des2, desc_size)
            out1[0][datum_pair_idx, ...] = scores1
            out1[1][datum_pair_idx, ...] = coords1
            out1[2][datum_pair_idx, ...] = descs1
            out2[0][datum_pair_idx, ...] = scores2
            out2[1][datum_pair_idx, ...] = coords2
            out2[2][datum_pair_idx, ...] = descs2
        # batch.image2 = batch.image1
        # batch.params2 = batch.params1
        # batch.pose2 = batch.pose1
        det_eval = compute_repeatability_batch(
            out1, out2, batch, cell=conv_size, matching_threshold=matching_threshold
        )
        des_eval = compute_matching_score_batch(
            out1, out2, batch, matching_threshold=matching_threshold
        )
        recall = compute_recall_batch(out1, out2, batch, conv_size=conv_size)
        recall_eval.append(recall)
        detector_eval.append(det_eval)
        descriptor_eval.append(des_eval)

    rs = [det[2] for det in detector_eval]
    le = [det[3] for det in detector_eval]
    print(
        f"Metrics for {model_name} with {nfeatures} features and matching threshold of {matching_threshold}"
    )
    print(f"Average rs: {np.array(rs).mean()}")
    print(f"Average le: {np.array(le).mean()}")
    print(f"Average ms: {np.array(descriptor_eval).mean()}")
    print(f"Average recall: {np.array(recall_eval).mean()}")


@app.command()
def main(
    config_file: str = "sips/configs/v0_dummy.yaml",
    nfeatures: int = 20000,
    model_name: str = "ORB",
    conv_size: int = 8,
    matching_threshold: int = 3,
):
    config = parse_train_file(config_file)
    conv_size = config.datasets.conv_size
    seed_everything(config.arch.seed, workers=True)
    dm = SonarDataModule(config.datasets)
    dm.prepare_data()
    dm.setup("validate")

    dataloader = dm.test_dataloader()

    evaluate(model_name, dataloader, nfeatures, matching_threshold, conv_size)


if __name__ == "__main__":
    app()
