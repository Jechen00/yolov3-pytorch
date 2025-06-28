#####################################
# Imports & Dependencies
#####################################
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision.ops import box_convert
from torchvision.tv_tensors import BoundingBoxes

import os
import textwrap
import subprocess
from PIL import Image
from abc import ABC, abstractmethod
from urllib.parse import urlparse
import seaborn as sns
import random
from typing import Tuple, List, Union, Callable, Optional

from src import evaluate
from src.data_setup import transforms
from src.utils import misc, convert
from src.utils.constants import BOLD_START, BOLD_END


#####################################
# Functions
#####################################
def download_with_curl(url: str, download_dir: str, unzip: bool = False):
    # Make sure the download directory exists
    os.makedirs(download_dir, exist_ok = True)
    
    parsed_url = urlparse(url)
    filename = os.path.basename(parsed_url.path) # Get .zip filename
    
    dest_path = os.path.join(download_dir, filename) # Path to download destination
    
    # Run curl to download the file
    if not os.path.exists(dest_path):
        print(f'{BOLD_START}[ALERT]{BOLD_END} Downloading {url}')
        subprocess.run(['curl', '-o', dest_path, url], check = True)
    else:
        print(f'{BOLD_START}[ALERT]{BOLD_END} File from {url} already downloaded at {dest_path}')
    
    if unzip:
        print(f'{BOLD_START}[ALERT]{BOLD_END} Extracting to {download_dir}')
        subprocess.run(['unzip', '-q', '-o', dest_path, '-d', download_dir], check = True)
        subprocess.run(['rm', dest_path], check = True)
        
    print(f'{BOLD_START}[ALERT]{BOLD_END} Download and/or extraction complete')

def load_classes(
        label_path: str, 
        return_idx_map: bool = True,
        clr_shuffle_seed: Optional[int] = None
) -> Union[Tuple[list, list], Tuple[list, list, dict, dict]]:
    '''
    Loads the class names and colors of a dataset using a `.names` file located at `label_path`.

    Args:
        label_path (str): The path to a `.names` file to load class labels.
        return_idx_maps (bool): Whether to return an additional dictionary
                                for mapping class labels to indices.
                                Default is True.
        clr_shuffle_seed (optional, int): A random seed used to shuffle the class colors from `sns.color_palette`.
                                          If not provided, colors are not shuffled. Default is None.

    Returns:
        class_names (list): List of class labels.
        class_clrs (list): List of RGB color tuples from `sns.color_palette` in normalized float format.

        If `return_idx_maps = True`:
            class_to_idx (dict): Dictionary mapping class labels to a unique index.
            idx_to_class (dict): Dictionary mapping unique indices to their class labels.
    '''
    with open(label_path, 'r') as f:
        class_names = [line.strip() for line in f if line.strip()]
    
    class_clrs = list(sns.color_palette(palette = 'hls', n_colors = len(class_names)))

    if clr_shuffle_seed is not None:
        random.Random(clr_shuffle_seed).shuffle(class_clrs)

    if return_idx_map:
        class_to_idx = {cls: i for i, cls in enumerate(class_names)}
        return class_names, class_clrs, class_to_idx
    
    else:
        return class_names, class_clrs
    

#####################################
# Classes
#####################################
class DetectionDatasetBase(ABC, Dataset):
    '''
    Mix-in class that provides common functions for dataset classes.
    This is not meant to be instantiated alone.

    scale_anchors (List[Tensor]): List of anchor tensors, one per scale. Each tensor has shape (num_anchors, 2),
                                  where the last dimension gives the (width, height) of the anchor 
                                  in units of the input size (pixels).    
    '''
    def __init__(self, 
                 root: str,
                 scale_anchors: List[torch.Tensor],
                 input_size: Union[int, Tuple[int, int]],
                 strides: List[Union[int, Tuple[int, int]]],
                 ignore_threshold: float,
                 label_path: str, 
                 dataset_name: str,
                 single_augs: Optional[Callable] = None,
                 mosaic_augs: Optional[Callable] = None,
                 mosaic_prob: float = 0.0,
                 min_box_scale: float = 0.01):
        super().__init__()

        self.root = root
        self.scale_anchors = scale_anchors
        self.input_size = input_size
        self.strides = [misc.make_tuple(stride) for stride in strides]
        self.ignore_threshold = ignore_threshold
        self.label_path = label_path
        self.dataset_name = dataset_name
        self.single_augs = single_augs
        self.mosaic_augs = mosaic_augs
        self.mosaic_prob = mosaic_prob
        self.min_box_scale = min_box_scale

        self.mosaic_resize = v2.Resize(size = 0) # 0 is just a placeholder
        self.mosaic_to_tensor = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale = True)
        ])

        self.class_names, self.class_clrs, self.class_to_idx = load_classes(self.label_path)
        self.set_input_size(input_size) # Set attributes for input_size, fmap_sizes, and single_resize
        self.anchors_info = self._get_anchors_info() # Dictionary with anchor/scale information for encoding
    
    def __getitem__(
            self, 
            idx: int
    ) -> Tuple[Union[Image.Image, torch.Tensor], List[torch.Tensor]]:
        '''
        Gets the transformed image and targets for a given index.

        Args:
            idx (int): Image index. For a mosaic, this refers to the index of the top-left image.

        Returns:
            img (Image.Image or torch.Tensor): The transformed image based on `idx`. 
                                               If `self.mosaic_prob > 0`, there is a `self.mosaic_prob * 100%` chance 
                                               that a mosaic of 4 images is returned.
            scale_targs (List[torch.Tensor]): List of encoded target tensors, one per scale of the model.
                                              Each has shape: (num_anchors, fmap_h, fmap_w, 5 + C)
        '''

        # P(X < x) = x for X ~ Unif[0, 1]
        if random.uniform(0, 1) < self.mosaic_prob:
            return self.load_mosaic_image_and_targets(idx)
        else:
            return self.load_single_image_and_targets(idx) 
        
    def __repr__(self) -> str:
        '''
        Return a readable string representation of the dataset.
        Includes dataset name, number of samples, number of classes, example image and target shapes,
        and whether images are transformed.
        '''
        examp_img, examp_targs = self.__getitem__(0)
        if isinstance(examp_img, torch.Tensor):
            img_shape = tuple(examp_img.shape)
        else:
            img_shape = 'N/A'

        targ_shapes = [tuple(targ.shape) for targ in examp_targs]

        dataset_str = f'''\
        {BOLD_START}Dataset:{BOLD_END} {self.dataset_name}
            {BOLD_START}Root location:{BOLD_END} {self.root}
            {BOLD_START}Number of samples:{BOLD_END} {self.__len__()}
            {BOLD_START}Number of classes:{BOLD_END} {len(self.class_names)}
            {BOLD_START}Strides:{BOLD_END} {self.strides}
            {BOLD_START}Image shape:{BOLD_END} {img_shape}
            {BOLD_START}Target shapes:{BOLD_END} {targ_shapes}
        '''
        return textwrap.dedent(dataset_str)
    
    def load_single_image_and_targets(
            self, 
            idx: int
    ) -> Tuple[Union[Image.Image, torch.Tensor], List[torch.Tensor]]:
        '''
        Loads a single transformed image and its targets, given an index.

        Args:
            idx (int): Image index.

        Returns:
            img (Image.Image or torch.Tensor): The transformed image at `idx`.
            scale_targs (List[torch.Tensor]): List of encoded target tensors, one per scale of the model.
                                              Each has shape: (num_anchors, fmap_h, fmap_w, 5 + C)
        '''
        img = self.get_img(idx)
        anno_info = self.get_anno_info(idx)

        if self.single_augs is not None:
            img, anno_info = self.single_augs(img, anno_info)

        img, anno_info = self.single_resize(img, anno_info)
        
        scale_targs = self._encode_yolov3_targets(anno_info)
        return img, scale_targs
    
    def load_mosaic_image_and_targets(
            self, 
            idx: int
    ) -> Tuple[Union[Image.Image, torch.Tensor], List[torch.Tensor]]:
        '''
        Loads a mosaic of 4 image along with their corresponding targets. 
        The top-left image is given by `idx`, while the rest of the 3 images are random chosen from the dataset.

        Reference: https://gmongaras.medium.com/yolox-explanation-mosaic-and-mixup-for-data-augmentation-3839465a3adf

        Args:
            idx (int): Image index for the top-left image of the mosaic.

        Returns:
            mosaic_img (Image.Image or torch.Tensor): The transformed mosaic image.
            scale_targs (List[torch.Tensor]): List of encoded target tensors, one per scale of the model.
                                              Each has shape: (num_anchors, fmap_h, fmap_w, 5 + C)
        '''
        input_h, input_w = self.input_size

        # List order: Top-Left, Top-Right, Bottom-Right, Bottom-Left
        all_img_idxs = [idx] + random.sample(range(self.__len__()), 3)
        all_bboxes, all_labels = [], []

        # -------------------
        # Creating Mosaic
        # -------------------
        mosaic_img = Image.new('RGB', (2 * input_w, 2 * input_h), color = (114, 114, 114))
        mosaic_positions = [(0, 0), (input_w, 0), (input_w, input_h), (0, input_h)]

        for img_idx, img_pos in zip(all_img_idxs, mosaic_positions):
            img = self.get_img(img_idx)
            anno_info = self.get_anno_info(img_idx)
            
            orig_w, orig_h = img.size
            lb_scale = min(input_h / orig_h, input_w / orig_w)
            scaled_w = int(orig_w * lb_scale)
            scaled_h = int(orig_h * lb_scale)
            
            # Update and resize image
            self.mosaic_resize.size = (scaled_h, scaled_w)
            img, anno_info = self.mosaic_resize(img, anno_info)
            
            # Paste image such that one of its inner corner aligns with canvas center
            paste_x, paste_y = img_pos

            if paste_x == 0:
                paste_x += input_w - scaled_w

            if paste_y == 0:
                paste_y += input_h - scaled_h
                
            mosaic_img.paste(img, (paste_x, paste_y))

            # Adjust bounding box positions
            bboxes = box_convert(
                anno_info['boxes'], 
                in_fmt = anno_info['boxes'].format.value.lower(),
                out_fmt = 'xyxy'
            ).data

            bboxes[:, ::2] += paste_x
            bboxes[:, 1::2] += paste_y

            all_bboxes.append(bboxes) # List of tensors of shape (num_objects, 4) in XYXY format
            all_labels.append(anno_info['labels']) # List of tensors of shape (num_objects,)

        all_bboxes = torch.concat(all_bboxes, dim = 0)
        all_labels = torch.concat(all_labels)

        # -------------------
        # Cropping Mosaic
        # -------------------
        # (crop_cx, crop_cy) is a random position chosen 
            # within a (input_w/4)x(input_h/4) box centered on the mosaic image
            # This ensures that the final cropped image will always contain a part of all 4 images
        crop_cx = random.randint(7 * input_w // 8, 9 * input_w // 8)
        crop_cy = random.randint(7 * input_h // 8, 9 * input_h // 8)

        # Crop region
        crop_xmin = crop_cx - input_w // 2
        crop_ymin = crop_cy - input_h // 2
        crop_xmax = crop_xmin + input_w
        crop_ymax = crop_ymin + input_h

        mosaic_img = mosaic_img.crop((crop_xmin, crop_ymin, crop_xmax, crop_ymax))

        # Shift and clamp bboxes to image boundaries
        all_bboxes[:, ::2] = (all_bboxes[:, ::2] - crop_xmin).clamp(0, input_w)
        all_bboxes[:, 1::2] = (all_bboxes[:, 1::2] - crop_ymin).clamp(0, input_h)

        # Remove bboxes that are completely outside of boundaries (i.e. those with no width/height post-clamp)
        valid_mask = ((all_bboxes[:, 2] - all_bboxes[:, 0] > 3) & 
                      (all_bboxes[:, 3] - all_bboxes[:, 1] > 3))
        
        # Create mosaic annotation info in a format suitable for v2 transforms
        mosaic_anno_info = {
            'boxes': BoundingBoxes(
                data = all_bboxes[valid_mask],
                canvas_size = (input_h, input_w),
                format = 'XYXY'
            ),
            'labels': all_labels[valid_mask]
        }

        if self.mosaic_augs is not None:
            mosaic_img, mosaic_anno_info = self.mosaic_augs(mosaic_img, mosaic_anno_info)

        mosaic_img = self.mosaic_to_tensor(mosaic_img)
        scale_targs = self._encode_yolov3_targets(mosaic_anno_info)
        
        return mosaic_img, scale_targs
    
    def _get_anchors_info(self):
        anchor_scale_idxs, anchors_per_scale = [], []
        for idx, anchors in enumerate(self.scale_anchors):
            num_anchors = len(anchors)
            anchor_scale_idxs.extend([idx for _ in range(num_anchors)])
            anchors_per_scale.append(num_anchors)
        
        anchors_info = {
            'anchors_wh': torch.concat(self.scale_anchors, dim = 0), # Shape: (num_tot_anchors, 2)
            'num_tot_anchors': sum(anchors_per_scale),
            'anchor_scale_idxs': torch.tensor(anchor_scale_idxs),
            'anchors_per_scale': torch.tensor(anchors_per_scale)
        }
        
        return anchors_info
    
    def set_input_size(self, input_size: Union[int, Tuple[int, int]]):
        '''
        Changes the size used to resize transformed images.
        '''
        self.input_size = misc.make_tuple(input_size)
        self.fmap_sizes = [(self.input_size[0]//s[0], self.input_size[1]//s[1]) 
                           for s in self.strides]

        self.single_resize = v2.Compose([
            v2.ToImage(), # Convert to Tensor
            transforms.LetterBox(size = self.input_size, fill = 114), # Resizes to input_size (preserves aspect ratio)
            v2.ToDtype(torch.float32, scale = True) # Rescales to [0, 1]
        ])

    def _encode_yolov3_targets(self, anno_info: dict) -> List[torch.Tensor]:
        anchors_wh = self.anchors_info['anchors_wh']
        num_tot_anchors = self.anchors_info['num_tot_anchors']
        anchor_scale_idxs = self.anchors_info['anchor_scale_idxs']
        anchors_per_scale = self.anchors_info['anchors_per_scale']

        scale_targs = [] # List of target tensors, one per scale
        for size, anchors in zip(self.fmap_sizes, self.scale_anchors):
            scale_targs.append(
                torch.zeros(anchors.shape[0], size[0], size[1], 5 + len(self.class_names))
            )

        labels = anno_info['labels']

        # Shape of bboxes_center and bboxes_corner: (num_bboxes, 4)
        # bboxes_center and bboxes_corner should be in units of the input size (pixel)
        base_format = anno_info['boxes'].format.value.lower()
        bboxes_corner = box_convert(
            boxes = anno_info['boxes'], 
            in_fmt = base_format,
            out_fmt = 'xyxy'
        ).data

        # # Clamp to ensure all bboxes are within image 
        # bboxes_corner[:, ::2] = bboxes_corner[:, ::2].clamp(0, img_w)
        # bboxes_corner[:, 1::2] = bboxes_corner[:, 1::2].clamp(0, img_h)
        
        bboxes_center = convert.corner_to_center_format(bboxes_corner)

        # Filter out objects that may have been too cut off due to transforms
        min_size_tensor = self.min_box_scale * torch.tensor(self.input_size).flip(dims = [0])
        valid_mask = (bboxes_center[:, 2:] > min_size_tensor).all(dim = 1)

        bboxes_center = bboxes_center[valid_mask]
        bboxes_corner = bboxes_corner[valid_mask]
        labels = labels[valid_mask]

        # Loop over objects in the image and assign them to an anchor/scale
        for label, bbox_center, bbox_corner in zip(labels, bboxes_center, bboxes_corner):
            cx, cy, w, h = bbox_center

            cxcy = bbox_center[:2].repeat(num_tot_anchors, 1) # Shape: (num_tot_anchors, 2)
            bbox_anchors = torch.concat([cxcy, anchors_wh], dim = -1) # Shape: (num_tot_anchors, 4)
            bbox_anchors = convert.center_to_corner_format(bbox_anchors) # Convert CXCYWH to XYXY

            # -------------------------------
            # Assign Responsible Anchor Box
            # -------------------------------
            ious = evaluate.calc_ious(bbox_corner.unsqueeze(0), bbox_anchors).squeeze() # Shape: (num_tot_anchors,)
            max_idx = ious.argmax(dim = -1).item()
            scale_idx = anchor_scale_idxs[max_idx]
            anchor_idx = max_idx - anchors_per_scale[:scale_idx].sum()

            fmap_h, fmap_w = self.fmap_sizes[scale_idx] # Height, width of the fmap
            stride_h, stride_w = self.strides[scale_idx] # Height, width of a fmap grid cell
            anchor_w, anchor_h = self.scale_anchors[scale_idx][anchor_idx] # Width, height of responsible anchor
            
            # Note: Although all_cxcy < img_wh ensures bbox centers are within image bounds,
                # due to integer division (i.e. flooring), grid indices (grid_x, grid_y) 
                # can sometimes equal fmap size (fmap_w, fmap_h), causing out-of-bounds errors.
            # Clamping to fmap_size - 1 ensures indices stay within valid feature map range ([0, fmap_size - 1]).
            grid_x = min(int(cx / stride_w), fmap_w - 1)
            grid_y = min(int(cy / stride_h), fmap_h - 1)

            targ_x = (cx / stride_w) - grid_x # x-offset within the cell (in [0, 1])
            targ_y = (cy / stride_h) - grid_y # y-offset within the cell (in [0, 1])

            targ_w = torch.log(w / anchor_w) # normalized width in log-scale
            targ_h = torch.log(h / anchor_h) # normalized height in log-scale

            # The 1 in the last element indicates object presence
            targ_bbox = torch.tensor([targ_x, targ_y, targ_w, targ_h, 1], dtype = torch.float32)
            scale_targs[scale_idx][anchor_idx, grid_y, grid_x, :5] = targ_bbox

            # One-hot encode the class label
            scale_targs[scale_idx][anchor_idx, grid_y, grid_x, 5 + label] = 1

            # -------------------------------
            # Assign Ignore Anchor Boxes
            # -------------------------------
            ignore_mask = (ious >= self.ignore_threshold)
            ignore_mask[max_idx] = False
            ignore_idxs = torch.where(ignore_mask)[0]

            for ig_idx in ignore_idxs:
                ig_scale_idx = anchor_scale_idxs[ig_idx]
                ig_anchor_idx = ig_idx - anchors_per_scale[:ig_scale_idx].sum()
                
                ig_fmap_h, ig_fmap_w = self.fmap_sizes[ig_scale_idx]
                ig_stride_h, ig_stride_w = self.strides[ig_scale_idx]

                ig_grid_x = min(int(cx / ig_stride_w), ig_fmap_w - 1)
                ig_grid_y = min(int(cy / ig_stride_h), ig_fmap_h - 1)
                                
                # Object score of -1 implies ignore this cell during loss calculation
                if scale_targs[ig_scale_idx][ig_anchor_idx, ig_grid_y, ig_grid_x, 4] == 0:
                    scale_targs[ig_scale_idx][ig_anchor_idx, ig_grid_y, ig_grid_x, 4] = -1
            
        return scale_targs
    
    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def get_img(self, idx: int) -> Image.Image:
        pass

    @abstractmethod
    def get_anno_info(self, idx: int) -> dict:
        pass