#####################################
# Imports & Dependencies
#####################################
import torch
from torchvision.transforms import v2
import torchvision.transforms.v2.functional as F
from torchvision.ops import box_convert
from torchvision.tv_tensors import BoundingBoxes

from PIL import Image
from typing import Tuple, Union, Optional, Literal

from src.utils import misc


#####################################
# Functions
#####################################
def get_single_transforms(train: bool = True,
                          aug_only: bool = False,
                          size: Union[int, Tuple[int, int]] = (416, 416)) -> Optional[v2.Compose]:
    '''
    Creates a torchvision transform pipeline for preprocessing a single image 
    during training or validation/testing. If performing multiscale training, 
    its recommended to set `aug_only=True` for the training transforms.

    Args:
        train (bool): If True, includes data augmentation transforms such as random HSV adjustments
                      and random affine transformations (scaling and translation). If False, only basic
                      ToImage and resizing transforms are applied.
                      Default is True.
        aug_only (bool): Whether to only return the data augmentation transforms. 
                         If `aug_only=True` and `train=False`, this function returns `None`. \
                         Default is False.
        size (int or Tuple[int, int]): The size to resize the input image into. 
                                       If type is `int`, it is assumed that resize will be square.
                                       This is only used if `resize = True`. Default is (416, 416).
              
    Returns:
        v2.Compose: The transform pipeline to be used in datasets.
    '''
    if train:
        transforms = [
            v2.ColorJitter(
                brightness = 0.5,
                saturation = 0.4,
                hue = 0.02
            ),
            v2.RandomGrayscale(p = 0.05),
            v2.RandomIoUCrop(
                min_scale = 0.5,
                min_aspect_ratio = 0.75,
                max_aspect_ratio = 1.33,
                sampler_options = [0.3, 0.4, 0.5, 0.7, 0.9, 1.1],
                trials = 20
            ),
            v2.RandomHorizontalFlip(p = 0.5),
            v2.RandomAffine(
                degrees = 10,
                scale = (0.9, 1.1),
                translate = (0.1, 0.1),
                fill = 114
            ),
            v2.RandomApply([v2.GaussianBlur(kernel_size = 3, sigma = (0.1, 1.5))], p = 0.1)
        ]
    else:
        transforms = []

    if not aug_only:
        transforms += [
            v2.ToImage(),
            LetterBox(size = size, fill = 114),
            v2.ToDtype(torch.float32, scale = True)
        ]

    compose_transforms = v2.Compose(transforms) if transforms else None
    return compose_transforms

def get_post_multi_transforms(aug_only: bool = False) -> Optional[v2.Compose]:
    '''
    Creates a torchvision transform pipeline for preprocessing a target image 
    **after** a multi-image augmentation (e.g. mosaic or mix-up).
    This includes data augmentations and ToImage transforms. 
    It is also assumed that any resizing is performed during the creation of the target image.

    Args:
        aug_only (bool): Whether to only return the data augmentation transforms. Default is False.
              
    Returns:
        v2.Compose: The transform pipeline to be used in datasets.
    '''
    transforms = [
        v2.ColorJitter(
            brightness = 0.5,
            saturation = 0.4,
            hue = 0.02
        ),
        v2.RandomHorizontalFlip(p = 0.5)
    ]

    if not aug_only:
        transforms += [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale = True)
        ]

    compose_transforms = v2.Compose(transforms) if transforms else None
    return compose_transforms

def remove_letterbox_pad(
    img: Union[torch.Tensor, Image.Image], 
    bboxes: torch.Tensor, 
    orig_size: Tuple[int, int]
) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    img (torch.Tensor or Image.Image): If a tensor, shape is (..., height, width).
    bboxes shape: (num_bboxes, 4) w/ last dim in XYXY
    orig_size: (height, width)
    '''
    if isinstance(img, torch.Tensor):
        lb_h, lb_w = img.shape[-2:]
    elif isinstance(img, Image.Image):
        lb_w, lb_h = img.size
    else:
        raise TypeError('`img` must be a Tensor or PIL Image')
    
    lb_scale = min(lb_h / orig_size[0], lb_w / orig_size[1])
    scaled_h = int(orig_size[0] * lb_scale)
    scaled_w = int(orig_size[1] * lb_scale)
    
    pad_h = lb_h - scaled_h
    pad_w = lb_w - scaled_w
    
    pad_l = pad_w // 2
    pad_t = pad_h // 2

    # Remove padding
    pad_rm_img = F.center_crop(img, output_size = (scaled_h, scaled_w))
    
    # Shift bboxes to new coordinates
    pad_rm_bboxes = bboxes.clone()
    pad_rm_bboxes[:, ::2] = (pad_rm_bboxes[:, ::2] - pad_l).clamp(0, 0.995 * scaled_w)
    pad_rm_bboxes[:, 1::2] = (pad_rm_bboxes[:, 1::2] - pad_t).clamp(0, 0.995 * scaled_h)
    
    return pad_rm_img, pad_rm_bboxes

def functional_letterbox(
    img: Union[Image.Image, torch.Tensor],
    size: Union[int, Tuple[int, int]], 
    anno_info: Optional[dict] = None, 
    fill: Union[int, Tuple[int, int, int]] = 0,
    return_bbox_fmt: Literal['orig', 'xyxy', 'cxcywh'] = 'orig'
) -> Tuple[Union[Image.Image, torch.Tensor], Optional[dict]]:
    
    size = misc.make_tuple(size) # (height, width)
    if isinstance(img, torch.Tensor):
        orig_h, orig_w = img.shape[-2:]
    elif isinstance(img, Image.Image):
        orig_w, orig_h = img.size
    else:
        raise TypeError('`img` must be a Tensor or PIL Image')
    
    # ------------
    # Resizing
    # ------------
    lb_scale = min(size[1] / orig_w, size[0] / orig_h)
    scaled_w = int(orig_w * lb_scale)
    scaled_h = int(orig_h * lb_scale)
    if anno_info is not None:
        orig_fmt = anno_info['boxes'].format.value.lower()

        img, anno_info = functional_resize(
            img = img, anno_info = anno_info,
            size = (scaled_h, scaled_w), return_bbox_fmt = 'xyxy'
        )
    else:
        img = F.resize(img, size = (scaled_h, scaled_w))

    # ------------
    # Padding
    # ------------
    pad_w = size[1] - scaled_w
    pad_h = size[0] - scaled_h
    pad_l = pad_w // 2
    pad_r = pad_w - pad_l
    pad_t = pad_h // 2
    pad_b = pad_h - pad_t

    img = F.pad(img, padding = (pad_l, pad_t, pad_r, pad_b), 
                fill = fill, padding_mode = 'constant')

    if anno_info is not None:
        xyxy_bboxes = anno_info['boxes'] # These are in XYXY format b/c functional_resize
        xyxy_bboxes[:, ::2] += pad_l
        xyxy_bboxes[:, 1::2] += pad_t
        
        # Decide output format
        if return_bbox_fmt == 'xyxy':
            out_bboxes = xyxy_bboxes
            out_fmt = 'xyxy'
        else:
            out_fmt = orig_fmt if return_bbox_fmt == 'orig' else return_bbox_fmt
            out_bboxes = box_convert(
                boxes = xyxy_bboxes, in_fmt = 'xyxy', out_fmt = out_fmt
            )
            
        anno_info['boxes'] = BoundingBoxes(data = out_bboxes,
                                           format = out_fmt.upper(),
                                           canvas_size = size)
        return img, anno_info
    else:
        return img
        
def functional_resize(
    img: Union[Image.Image, torch.Tensor],
    size: Union[int, Tuple[int, int]], 
    anno_info: Optional[dict] = None, 
    return_bbox_fmt: Literal['orig', 'xyxy', 'cxcywh'] = 'orig'
) -> Tuple[Union[Image.Image, torch.Tensor], Optional[dict]]:
    
    size = misc.make_tuple(size) # (height, width)
    if isinstance(img, torch.Tensor):
        orig_h, orig_w = img.shape[-2:]
    elif isinstance(img, Image.Image):
        orig_w, orig_h = img.size
    else:
        raise TypeError('`img` must be a Tensor or PIL Image')
    
    img = F.resize(img, size = size)
    if anno_info is not None:
        orig_fmt = anno_info['boxes'].format.value.lower()

        # Convert to XYXY format
        xyxy_bboxes = box_convert(
            boxes = anno_info['boxes'], in_fmt = orig_fmt, out_fmt = 'xyxy'
        )

        xyxy_bboxes[:, ::2] *= size[1] / orig_w
        xyxy_bboxes[:, 1::2] *= size[0] / orig_h

        # Decide output format
        if return_bbox_fmt == 'xyxy':
            out_bboxes = xyxy_bboxes
            out_fmt = 'xyxy'
        else:
            out_fmt = orig_fmt if return_bbox_fmt == 'orig' else return_bbox_fmt
            out_bboxes = box_convert(
                boxes = xyxy_bboxes, in_fmt = 'xyxy', out_fmt = out_fmt
            )

        anno_info = anno_info.copy()
        anno_info['boxes'] = BoundingBoxes(data = out_bboxes,
                                           format = out_fmt.upper(),
                                           canvas_size = size)
        return img, anno_info
    else:
        return img


#####################################
# Classes
#####################################
class LetterBox():
    def __init__(self, 
                 size: Union[int, Tuple[int, int]], 
                 fill: Union[int, Tuple[int, int, int]] = 0):
        self.size = misc.make_tuple(size) # Height, width
        self.fill = fill
        
        self.resize_transform = v2.Resize(size = (0, 0))
        self.pad_transform = v2.Pad(padding = 0, fill = self.fill, padding_mode = 'constant')
        
    def __call__(
        self, 
        img: Union[Image.Image, torch.Tensor], 
        anno_info: Optional[dict] = None
    ) -> Tuple[Union[Image.Image, torch.Tensor], Optional[dict]]:
        
        if isinstance(img, torch.Tensor):
            orig_h, orig_w = img.shape[-2:]
        elif isinstance(img, Image.Image):
            orig_w, orig_h = img.size
        else:
            raise TypeError('`img` must be a Tensor or PIL Image')

        # Resizing scale
        lb_scale = min(self.size[1] / orig_w, self.size[0] / orig_h)
        scaled_w = int(orig_w * lb_scale)
        scaled_h = int(orig_h * lb_scale)

        # Padding after resize
        pad_w = self.size[1] - scaled_w
        pad_h = self.size[0] - scaled_h
        pad_l = pad_w // 2
        pad_r = pad_w - pad_l
        pad_t = pad_h // 2
        pad_b = pad_h - pad_t
        
        # Update transforms
        self.resize_transform.size = (scaled_h, scaled_w)
        self.pad_transform.padding = (pad_l, pad_t, pad_r, pad_b)
        
        # Resize -> Pad
        img, anno_info = self.resize_transform(img, anno_info)
        img, anno_info = self.pad_transform(img, anno_info)
        
        if anno_info is not None:
            return img, anno_info
        else:
            return img