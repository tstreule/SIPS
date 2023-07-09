import cv2 as cv
import numpy as np
import numpy.typing as npt
import torch
import tqdm
from matplotlib import pyplot as plt
from pytorch_lightning import seed_everything
from torch import nn

from sips.configs.parsing import parse_train_file
from sips.data_extraction.plotting import plot_overlap
from sips.datasets.data_module import SonarDataModule
from sips.evaluation.descriptor_evaluation import compute_matching_score_batch
from sips.evaluation.detector_evaluation import compute_repeatability_batch
from sips.utils.keypoint_matching import _unravel_index_2d


def format_scores_coords_descs(
    kps: tuple[cv.KeyPoint],
    desc: npt.NDArray[np.uint8],
    desc_size: int,
    conv_size: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Create a fake image with the highest orb response at each pixel, if no
    # key point was found at a pixel leave at -1000. Note that ORB finds
    # subpixel-level keypoints, by casting to int we ensure that all keypoints
    # within the same pixel correspond to that pixel.
    kp_orb1_uv = torch.full((512, 512), -1000).to(torch.float)
    kp_pt = torch.full((2, 512, 512), -1).to(torch.float)
    kp_desc = torch.full((desc_size, 512, 512), -1).to(torch.float)
    for idx, kp in enumerate(kps):
        # If the value at a pixel is bigger than -1000 that means that we already
        # stored a response at that pixel. Keep the bigger response
        if kp_orb1_uv[int(kp.pt[0]), int(kp.pt[1])] > -1000:
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
    mask = pool > -1000
    indices_ = indices[mask]
    row, col = _unravel_index_2d(indices_, [4096, 512])
    scores = torch.full((1, 64, 64), torch.nan)
    coords = torch.full((2, 64, 64), torch.nan)
    descs = torch.full((desc_size, 64, 64), torch.nan)
    scores[mask] = kp_orb1_uv[row, col]
    coords[:, mask.squeeze()] = kp_pt[:, row, col]
    descs[:, mask.squeeze()] = kp_desc[:, row, col]
    return scores, coords, descs


def main(plot: bool = True, conv_size: int = 8):
    config_file = "sips/configs/v0_dummy.yaml"
    config = parse_train_file(config_file)
    seed_everything(config.arch.seed, workers=True)
    dm = SonarDataModule(config.datasets)
    dm.prepare_data()
    dm.setup("validate")
    orb = cv.ORB_create(10000)
    val_dataloader = dm.val_dataloader()
    detector_eval = []
    descriptor_eval = []
    for batch in iter(val_dataloader):
        out1 = (
            torch.full((batch.batch_size, 1, 64, 64), torch.nan),
            torch.full((batch.batch_size, 2, 64, 64), torch.nan),
            torch.full((batch.batch_size, 32, 64, 64), torch.nan),
        )
        out2 = (
            torch.full((batch.batch_size, 1, 64, 64), torch.nan),
            torch.full((batch.batch_size, 2, 64, 64), torch.nan),
            torch.full((batch.batch_size, 32, 64, 64), torch.nan),
        )
        for datum_pair_idx in range(batch.batch_size):
            im1 = batch.image1[datum_pair_idx, ...]
            im2 = batch.image1[datum_pair_idx, ...]
            kp1, des1 = orb.detectAndCompute(np.array(im1.squeeze()), None)
            kp2, des2 = orb.detectAndCompute(np.array(im2.squeeze()), None)
            scores1, coords1, descs1 = format_scores_coords_descs(kp1, des1, 32)
            scores2, coords2, descs2 = format_scores_coords_descs(kp2, des2, 32)
            out1[0][datum_pair_idx, ...] = scores1
            out1[1][datum_pair_idx, ...] = coords1
            out1[2][datum_pair_idx, ...] = descs1
            out2[0][datum_pair_idx, ...] = scores2
            out2[1][datum_pair_idx, ...] = coords2
            out2[2][datum_pair_idx, ...] = descs2
        batch.image2 = batch.image1
        batch.params2 = batch.params1
        batch.pose2 = batch.pose1
        det_eval = compute_repeatability_batch(
            out1, out2, batch, conv_size, matching_threshold=10
        )
        des_eval = compute_matching_score_batch(out1, out2, batch)
        detector_eval.append(det_eval)
        descriptor_eval.append(des_eval)
    rs = [det[2] for det in detector_eval]
    le = [det[3] for det in detector_eval]
    print(f"Average rs: {np.array(rs).mean()}")
    print(f"Average le: {np.array(le).mean()}")
    print(f"Average ms: {np.array(descriptor_eval).mean()}")
    print(f"detector evaluation lenght: {len(detector_eval)}")
    print(f"descriptor evaluation lenght: {len(descriptor_eval)}")
    for batch in iter(val_dataloader):
        im1 = batch.image1[1, ...]
        im2 = batch.image1[1, ...]
        pose1 = batch.pose1[1]
        pose2 = batch.pose1[1]
        params1 = batch.params1[1]
        params2 = batch.params1[1]
        # sift = cv.SIFT_create()
        kp1, des1 = orb.detectAndCompute(np.array(im1.squeeze()), None)
        kp2, des2 = orb.detectAndCompute(np.array(im2.squeeze()), None)
        scores1, coords1, descs1 = format_scores_coords_descs(kp1, des1, 32)
        scores2, coords2, descs2 = format_scores_coords_descs(kp2, des2, 32)

        from sips.utils.point_projection import warp_image_batch

        kp1_proj = warp_image_batch(
            coords1.unsqueeze(0), [params1], [pose1], [params2], [pose2], (512, 512)
        )
        kp2_proj = warp_image_batch(
            coords2.unsqueeze(0), [params2], [pose2], [params1], [pose1], (512, 512)
        )

        _, ax = plt.subplots(2, 2)
        ax[0, 0].imshow(im1.squeeze(), cmap="gray")
        ax[1, 0].imshow(im2.squeeze(), cmap="gray")
        ax[0, 1].imshow(im1.squeeze(), cmap="gray")
        ax[1, 1].imshow(im2.squeeze(), cmap="gray")
        ax[0, 1].plot(coords1[0].flatten(), coords1[1].flatten(), ".r")
        ax[1, 1].plot(coords2[0].flatten(), coords2[1].flatten(), ".r")
        ax[1, 1].plot(
            kp1_proj[:, :, 0, ...].flatten(),
            kp1_proj[:, :, 1].flatten(),
            ".g",
            alpha=0.5,
        )
        ax[0, 1].plot(
            kp2_proj[:, :, 0, ...].flatten(),
            kp2_proj[:, :, 1].flatten(),
            ".g",
            alpha=0.5,
        )
        plt.show()

    print()
    # im1 = cv.imread("data/filtered/boatpolice-001/images/sonar_1688385920067311431.png")
    # im1 = cv.resize(im1, (512, 512))
    # im2 = cv.imread("data/filtered/boatpolice-001/images/sonar_1688386288447353316.png")
    # im2 = cv.resize(im2, (512, 512))

    # kp_orb1, des_orb1 = orb.detectAndCompute(im1, None)
    # kp_orb2, des_orb2 = orb.detectAndCompute(im2, None)

    # orb1_scores, orb1_coords, orb1_descs = format_scores_coords_descs(
    #     kp_orb1, des_orb1, 32
    # )

    # if plot:
    #     plt.imshow(cv.drawKeypoints(im1, kp_orb1, None, color=(0, 255, 0), flags=0))
    #     plt.plot(orb1_coords[0].flatten(), orb1_coords[1].flatten(), ".r")
    #     plt.show()

    # sift = cv.SIFT_create()
    # kp_sift1, des_sift1 = sift.detectAndCompute(im1, None)
    # sift1_scores, sift1_coords, sift1_descs = format_scores_coords_descs(
    #     kp_sift1, des_sift1, 128
    # )
    # if plot:
    #     plt.imshow(cv.drawKeypoints(im1, kp_sift1, None, color=(0, 255, 0), flags=0))
    #     plt.plot(sift1_coords[0].flatten(), sift1_coords[1].flatten(), ".r")
    #     plt.show()

    # return


if __name__ == "__main__":
    main()
