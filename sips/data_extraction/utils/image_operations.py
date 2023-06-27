import torch
import torchvision.transforms as T
from PIL import Image, ImageFilter
from torch import nn

from sips.utils.keypoint_matching import _unravel_index_2d


# Transform PIL Image to torch tensor
def image_to_tensor(
    im: Image.Image, im_shape: tuple[int, int] = (512, 512)
) -> torch.Tensor:
    transform = T.Compose([T.PILToTensor()])
    return transform((im.resize(im_shape)))


# Blur image
def blur_image(im: Image.Image) -> Image.Image:
    return im.filter(ImageFilter.BLUR)


# Find pixels that are brighter than threshold. Only keep one per 8x8
# grid. Return as uv coordinates
def find_bright_spots(im: torch.Tensor, conv_size=8, bs_threshold=210) -> torch.Tensor:
    maxpool = nn.MaxPool2d(conv_size, stride=conv_size, return_indices=True)
    pool, indices = maxpool(im.to(torch.float))
    mask = pool > bs_threshold
    masked_indices = indices[mask]
    row, col = _unravel_index_2d(masked_indices, [4096, 512])
    bright_spots_temp = torch.column_stack((row, col)).to(torch.float)
    bright_spots = torch.full((64 * 64, 2), torch.nan)
    bright_spots[: bright_spots_temp.shape[0], :] = bright_spots_temp
    return bright_spots.permute(1, 0).view(2, 64, 64)
