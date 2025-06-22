#####################################
# Imports & Dependencies
#####################################
import torch
from torchvision.transforms import v2

import os
import textwrap
import subprocess
from PIL import Image
from abc import ABC, abstractmethod
from urllib.parse import urlparse
from typing import Tuple, List, Union

from src import evaluate
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


#####################################
# Classes
#####################################
class DetectionDatasetBase(ABC):
    '''
    Mix-in class that provides common functions for dataset classes.
    This is not meant to be instantiated alone.

    sscale_anchors (List[Tensor]): List of anchor tensors, one per scale. Each tensor has shape (num_anchors, 2),
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
                 dataset_name: str):
        super().__init__()

        self.root = root
        self.scale_anchors = scale_anchors
        self.input_size = input_size
        self.strides = [misc.make_tuple(stride) for stride in strides]
        self.ignore_threshold = ignore_threshold
        self.label_path = label_path
        self.dataset_name = dataset_name

        self.classes, self.class_to_idx, self.idx_to_class = self._load_class_names()
        self.set_input_size(input_size) # Set attributes for input_size, fmap_sizes, and resize_transforms
        self.anchors_info = self._get_anchors_info() # Dictionary with anchor/scale information for encoding
    
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
            {BOLD_START}Number of classes:{BOLD_END} {len(self.classes)}
            {BOLD_START}Strides:{BOLD_END} {self.strides}
            {BOLD_START}Image shape:{BOLD_END} {img_shape}
            {BOLD_START}Target shapes:{BOLD_END} {targ_shapes}
        '''
        return textwrap.dedent(dataset_str)
    
    def _load_class_names(self) -> Tuple[list, dict, dict]:
        '''
        Loads the class names of a dataset from a `.names` file located at `label_path`.

        Returns:
            classes (list): List of class labels.
            class_to_idx (dict): Dictionary mapping class labels to a unique index.
            idx_to_class (dict): Dictionary mapping unique indices to their class labels.
        '''
        with open(self.label_path, 'r') as f:
            classes = [line.strip() for line in f if line.strip()]
        
        class_to_idx = {cls: i for i, cls in enumerate(classes)}
        idx_to_class = {i: cls for i, cls in enumerate(classes)}
        
        return classes, class_to_idx, idx_to_class 
    
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
        Changes the resizing image size used in resize_transforms.
        '''
        self.input_size = misc.make_tuple(input_size)
        self.fmap_sizes = [(self.input_size[0]//s[0], self.input_size[1]//s[1]) 
                           for s in self.strides]
        
        self.resize_transforms = v2.Compose([
            v2.ToImage(), # Convert to Tensor
            v2.Resize(size = self.input_size), # Resizes to input_size
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
                torch.zeros(anchors.shape[0], size[0], size[1], 5 + len(self.classes))
            )

        img_wh = torch.tensor(anno_info['boxes'].canvas_size).flip(dims = [0])
        labels = anno_info['labels']
        base_format = anno_info['boxes'].format.value

        # Shape of bboxes_center and bboxes_corner: (num_bboxes, 4)
        # bboxes_center and bboxes_corner should be in units of the input size (pixel)
        if base_format == 'CXCYWH':
            bboxes_center = anno_info['boxes'].data.clone()
            bboxes_corner = convert.center_to_corner_format(bboxes_center) # Converts center (CXCYWH) to corner (XYXY)
        elif base_format == 'XYXY':
            bboxes_corner = anno_info['boxes'].data.clone()
            bboxes_center = convert.corner_to_center_format(bboxes_corner) # Converts corner (XYXY) to center (CXCYWH)
        elif base_format == 'XYWH':
            bboxes_center = anno_info['boxes'].data.clone()
            bboxes_center[:, :2] +=  bboxes_center[:, 2:] / 2 # Converts XYWH to center (CXCYWH)
            bboxes_corner = convert.center_to_corner_format(bboxes_center) # Converts center (CXCYWH) to corner (XYXY)
        else:
            raise ValueError("Format of bounding boxes in `anno_info` must be `'CXCYWH'` or `'XYXY'`")
        
        all_cxcy = bboxes_center[:, :2]
        all_wh = bboxes_center[:, 2:]

        # Filter out objects that may have been completely cut off due to transforms
        valid_mask = (
            (all_wh > 0).all(dim = 1) &
            (0 <= all_cxcy).all(dim = 1) &
            (all_cxcy < img_wh).all(dim = 1)
        )

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
    def __getitem__(
        self, 
        idx: int
    ) -> Tuple[Union[Image.Image, torch.Tensor], List[torch.Tensor]]:
        pass