#####################################
# Imports & Dependencies
#####################################
import torch
from torchvision.transforms import v2

from typing import Tuple, Union

from src.utils import misc


#####################################
# Functions
#####################################
def get_transforms(train: bool = True,
                   resize: bool = True,
                   to_float: bool = True,
                   size: Union[int, Tuple[int, int]] = (416, 416)) -> v2.Compose:
    '''
    Creates a torchvision transform pipeline for preprocessing images 
    during training or validation/testing. If performing multiscale training, 
    its recommended to set `resize = False` and `to_float = False`.

    Args:
        train (bool): If True, includes data augmentation transforms such as random HSV adjustments
                      and random affine transformations (scaling and translation). If False, only basic
                      ToImage and optionally resizing transforms are applied.
                      Default is True.
        resize (bool): Whether to include resizing in transforms. Default is True.
        to_float (bool): Whether to include transforming to float32.
                         If `to_float = True`, pixels will also be rescaled to [0, 1]. Default is True.
        size (int or Tuple[int, int]): The size to resize the input image into. 
                                       If type is `int`, it is assumed that resize will be square.
                                       This is only used if `resize = True`. Default is (416, 416).

        Note: If `resize = False` and `to_float = False`, the transforms will not include `v2.ToImage`.
              
    Returns:
        v2.Compose: The transform pipeline to be used in datasets.
    '''
    transforms = []
    if resize or to_float:
        transforms.append(v2.ToImage()) # Convert to tensor (dtype is usually uint8)

    if resize:
        size = misc.make_tuple(size)
        transforms.append(v2.Resize(size = size))

    if to_float:
        transforms.append(v2.ToDtype(torch.float32, scale = True)) # Rescales to [0, 1]

    if train:
        transforms = [
            v2.RandomIoUCrop(
                min_scale = 0.5,
                min_aspect_ratio = 0.5,
                max_aspect_ratio = 2.0,
                sampler_options = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1],
                trials = 20
            ),
            v2.RandomHorizontalFlip(p = 0.5),
            v2.RandomAffine(
                degrees = 5,
                scale = (0.8, 1.2),
                translate = (0.1, 0.1),
            ),
            v2.RandomPerspective(p = 0.2, distortion_scale = 0.08, fill = 114),
            v2.ColorJitter(
                brightness = 0.25,
                contrast = 0.15,
                saturation = 0.25,
                hue = 0.1  
            ),
            v2.RandomGrayscale(p = 0.1),
            v2.RandomAdjustSharpness(p = 0.2, sharpness_factor = 2)
        ] + transforms

    compose_transforms = v2.Compose(transforms) if transforms else None
    return compose_transforms