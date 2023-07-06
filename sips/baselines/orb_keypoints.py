import cv2 as cv
import numpy as np
import numpy.typing as npt
import torch
from matplotlib import pyplot as plt
from torch import nn

from sips.configs.parsing import parse_train_file
from sips.data_extraction.plotting import plot_overlap
from sips.utils.keypoint_matching import _unravel_index_2d


def bring_to_form(
    im: torch.Tensor,
    desc: npt.NDArray[np.uint8],
    desc_idx: torch.Tensor,
    kp_coords: torch.Tensor,
    conv_size=8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Find pixels that are brighter than threshold. Only keep one per 8x8
    grid. Return as uv coordinates

    Parameters
    ----------
    im : torch.Tensor
        input image
    conv_size : int, optional
        convolution size for maxpool, by default 8
    bs_threshold : int, optional
        bright spot threshold, by default 210

    Returns
    -------
    torch.Tensor
        bright spots with shape (2, 64, 64)
    """
    maxpool = nn.MaxPool2d(conv_size, stride=conv_size, return_indices=True)
    pool, indices = maxpool(im.to(torch.float))
    indices = indices.flatten()
    row, col = _unravel_index_2d(indices, [4096, 512])
    bright_spots_temp = torch.column_stack((col, row)).to(
        torch.float
    )  # Switched col and row in order to match setup within network
    bright_spots = torch.full((64 * 64, 2), torch.nan)
    bright_spots[: bright_spots_temp.shape[0], :] = bright_spots_temp
    bright_spots = bright_spots.permute(1, 0).view(2, 64, 64)
    # Set cells where max value equals to default value of -1000 to nan
    mask = pool == -1000
    pool[mask] = torch.nan
    bright_spots[0][mask.squeeze()] = torch.nan
    bright_spots[1][mask.squeeze()] = torch.nan
    return pool, bright_spots


def main(plot: bool = True):
    im1 = cv.imread("data/filtered/boatpolice-001/images/sonar_1688385920067311431.png")
    im2 = cv.imread("data/filtered/boatpolice-001/images/sonar_1688386288447353316.png")
    orb = cv.ORB_create(100)
    kp_orb1, des_orb1 = orb.detectAndCompute(im1, None)
    kp_orb2, des_orb2 = orb.detectAndCompute(im2, None)

    # Create a fake image with the highest orb response at each pixel, if no
    # key point was found at a pixel leave at -1000. Note that ORB finds
    # subpixel-level keypoints, by casting to int we ensure that all keypoints
    # within the same pixel correspond to that pixel.
    kp_orb1_uv = torch.full((512, 512), -1000).to(torch.float)
    kp_idx = torch.full((512, 512), -1).to(torch.int)
    kp_pt = torch.full((2, 512, 512), -1).to(torch.float)
    kp_desc = torch.full((32, 512, 512), -1).to(torch.float)
    for idx, kp in enumerate(kp_orb1):
        # If the value at a pixel is bigger than -1000 that means that we already
        # stored a response at that pixel. Keep the bigger response
        if kp_orb1_uv[int(kp.pt[0]), int(kp.pt[1])] > -1000:
            if kp_orb1_uv[int(kp.pt[0]), int(kp.pt[1])] > kp.response:
                continue
            else:
                kp_orb1_uv[int(kp.pt[0]), int(kp.pt[1])] = kp.response
                kp_idx[int(kp.pt[0]), int(kp.pt[1])] = idx
                kp_pt[0, int(kp.pt[0]), int(kp.pt[1])] = kp.pt[0]
                kp_pt[1, int(kp.pt[0]), int(kp.pt[1])] = kp.pt[1]
                kp_desc[:, int(kp.pt[0]), int(kp.pt[1])] = torch.tensor(des_orb1[idx])
        else:
            kp_orb1_uv[int(kp.pt[0]), int(kp.pt[1])] = kp.response
            kp_idx[int(kp.pt[0]), int(kp.pt[1])] = idx
            kp_pt[0, int(kp.pt[0]), int(kp.pt[1])] = kp.pt[0]
            kp_pt[1, int(kp.pt[0]), int(kp.pt[1])] = kp.pt[1]
            kp_desc[:, int(kp.pt[0]), int(kp.pt[1])] = torch.tensor(des_orb1[idx])
    kp_orb1_scores, kp_orb1_coords = bring_to_form(
        kp_orb1_uv.reshape((1, 512, 512)), des_orb1, kp_idx, kp_pt
    )
    maxpool = nn.MaxPool2d(8, stride=8, return_indices=True)
    pool, indices = maxpool(kp_orb1_uv.reshape((1, 512, 512)).to(torch.float))
    mask = pool > -1000
    indices_ = indices[mask]
    row, col = _unravel_index_2d(indices_, [4096, 512])
    scores = torch.full((1, 64, 64), torch.nan)
    coords = torch.full((2, 64, 64), torch.nan)
    descs = torch.full((32, 64, 64), torch.nan)
    scores[mask] = kp_orb1_uv[row, col]
    coords[:, mask.squeeze()] = kp_pt[:, row, col]
    descs[:, mask.squeeze()] = kp_desc[:, row, col]
    if plot:
        plt.imshow(cv.drawKeypoints(im1, kp_orb1, None, color=(0, 255, 0), flags=0))
        plt.plot(coords[0].flatten(), coords[1].flatten(), ".r")
        plt.show()
    return

    sift = cv.SIFT_create()
    kp_sift1, des_sift1 = sift.detectAndCompute(im1, None)
    _, ax = plt.subplots(2)
    # ax[0].imshow(cv.drawKeypoints(im1, kp_orb1, None, color=(0, 255, 0), flags=0))
    # ax[1].imshow(cv.drawKeypoints(im1, kp_sift1, None, color=(0, 255, 0), flags=0))
    config = parse_train_file("sips/configs/v0_dummy.yaml").datasets
    plot_overlap(config, config.rosbags[0], 0, False)

    return


if __name__ == "__main__":
    main()
