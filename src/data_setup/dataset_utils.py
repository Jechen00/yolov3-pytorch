#####################################
# Imports & Dependencies
#####################################
import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2.functional as F
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
from typing import Tuple, List, Union, Callable, Optional, Literal, Dict, Any

from src import evaluate
from src.data_setup import transforms
from src.utils import misc, convert
from src.utils.constants import BOLD_START, BOLD_END


#####################################
# Functions
#####################################
def download_with_curl(url: str, download_dir: str, unzip: bool = False):
    '''
    Uses curl to download a file from a URL to a directory and optionally unzips it.

    Args:
        url (str): The URL of the file to download.
        download_dir (str): TLocal directory to save the downloaded file.
        unzip (bool): Whether to unzip the downloaded file. 
                      This will break if the file is not a .zip file. Default is False.
    '''
    # Make sure the download directory exists
    os.makedirs(download_dir, exist_ok = True)
    
    parsed_url = urlparse(url)
    filename = os.path.basename(parsed_url.path) # Get filename
    
    dest_path = os.path.join(download_dir, filename) # Path to download destination
    
    # Run curl to download the file
    if not os.path.exists(dest_path):
        print(f'{BOLD_START}[ALERT]{BOLD_END} Downloading {url}')
        subprocess.run(['curl', '-L', '-o', dest_path, url], check = True)
    else:
        print(f'{BOLD_START}[ALERT]{BOLD_END} File from {url} already downloaded at {dest_path}')
    
    if unzip:
        if not filename.endswith('.zip'):
            raise ValueError(f'Cannot unzip: {filename} is not a .zip file')
        
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
    
def wh_k_means(bbox_whs: torch.Tensor, 
               k: int = 9, 
               max_iter: int = 1000, 
               update_method: Literal['mean', 'median'] = 'mean',
               round_whs: bool = True,
               area_order: Literal['desc', 'asc'] = 'asc',
               input_size: Tuple[int, int] = (416, 416)) -> torch.Tensor:
    '''
    Performs k-means clustering on a set of bounding box (width, height) pairs 
    to compute `k` representative anchor boxes (cluster centroids).
    This distance metric used is (1 - IoU), following typical YOLO models.

    Args:
        bbox_whs (torch.Tensor): Tensor of shape (num_bboxes, 2) containing (width, height) 
                                 pairs of bounding boxes, normalized to [0, 1].
        k (int): Number of anchor boxes (clusters) to compute. Default is 9.
        max_iter (int): Maximum number of iterations to run the k-means algorithm. Default is 1000.
        update_method (Literal['mean', 'median']): Method to update the cluster centroids.
                                                    - mean: Use the mean of the grouped bboxes.
                                                    - median: Use the median of the grouped bboxes.
        round_whs (bool): Whether to round the final anchor widths and heights to the nearest integers. Default is True.
        area_order (Literal['desc', 'asc']): The order, in terms of area, to return the anchors.
                                                - 'desc' refers to descending order of area.
                                                - 'asc' refers to ascending order of area.
                                             Default is 'asc'.
        input_size (Tuple[int, int]): Input size (height, width) used to scale the anchors from [0, 1] to pixels. 
                                      Default is (416, 416).

    Returns:
        torch.Tensor: An anchor tensor of shape (k, 2), where the last dimension gives the (width, height) of the anchor boxes.
                      The anchors are sorted in terms of area, with the order set by `area_order`.
    '''
    num_bboxes = bbox_whs.shape[0]
    anchors = bbox_whs[random.sample(range(num_bboxes), k)].clone() # Shape: (k, 2)
    grouped_whs = torch.full((num_bboxes, ), -1) # Initalize grouping
    
    # Perform K-means clustering
    for _ in range(max_iter):
        prev_grouped_whs = grouped_whs.clone()
        dist_ious = 1 - evaluate.calc_ious_wh(bbox_whs, anchors) # Shape: (num_bboxes, k)

        # Assign each bbox_wh to an anchor_wh group, based on minimum 1 - IoU
        grouped_whs = dist_ious.argmin(dim = -1) # Shape: (num_bboxes, )

        if torch.equal(prev_grouped_whs, grouped_whs):
            break

        # Update anchors by taking mean or median wh of each group
        for i in range(k):
            if (grouped_whs == i).sum() == 0:   
                continue # Skip empty groups

            if update_method == 'mean':
                anchors[i] = bbox_whs[grouped_whs == i].mean(dim = 0)

            elif update_method == 'median':
                anchors[i] = bbox_whs[grouped_whs == i].median(dim = 0).values
               
    # Get anchors in units of the input size (pixels)
    anchors[:, 0] *= input_size[1]
    anchors[:, 1] *= input_size[0]
    
    # Sort by area
    argsort_descending = (area_order == 'desc')
    anchors = anchors[anchors.prod(dim = -1).argsort(descending = argsort_descending)]
    
    if round_whs:
        anchors = anchors.round()
        
    return anchors.float() # Shape: (k, 2)


#####################################
# Classes
#####################################
class DetectionDatasetBase(ABC, Dataset):
    '''
    Mix-in class that provides common functions for object detection dataset classes.
    This is not meant to be instantiated alone.
    
    Args:
        root (str): Root directory where the dataset should be stored or loaded from.
        label_path (str): The path to a `.names` file to load class labels.
        scale_anchors (List[torch.tensor]): List of anchor tensors for each scale of the model.
                                            Each element has shape: (num_anchors, 2), where the last dimension gives 
                                            the (width, height) of the anchor in units of the input size (pixels).
        strides (List[Union[int, Tuple[int, int]]]): List of strides corresponding to each scale of the model.
                                                     If an element is a `tuple`, it should refer to the strides for (height, width).
                                                     If an element is an `int`, it is assumed that the stride 
                                                     is the same for height and width.
        default_input_size (Union[int, Tuple[int, int]]): Default input size used to resize images in `__getitem__` 
                                                          when no specific size is provided.
                                                          If a `tuple`, it should be (height, width). 
                                                          If an `int`, it is assumed to be square.
        ignore_threshold (float): IoU threshold used during target encoding to determine which anchors should be ignored.
                                  Anchors with an IoU greater than `ignore_threshold`, but are not the best matching anchor for an object 
                                  will be ignored in loss calculations (they are not treated as negatives and have P(object) = -1).
                                  Default is 0.5.
        single_augs (optional, Callable): Data augmentations to apply to the image in `__getitem__` 
                                          when only a single image is returned (i.e., no multi-image augmentations).
        multi_augs (Literal['mosaic', 'mixup'] or List[Literal['mosaic', 'mixup']]): 
                        The type(s) of multi-image augmentation to apply when combining images in `__getitem__`. 
                        If a string is passed (e.g., `'mosaic'` or `'mixup'`), that augmentation is always applied.  
                        If a list is passed (e.g. ['mosaic', 'mixup']), one augmentation is uniformly sampled 
                        each time a multi-image augmentation is applied. Default is 'mosaic'.
        post_multi_augs (optional, Callable): Data augmentation to apply after a multi-image augmentation is performed.
        multi_aug_prob (float): The probability of applying a multi-image augmentation. Default is 0.0 (no multi-image augmentations).
        mixup_alpha (float): Alpha parameter used for the Beta(alpha, alpha) distribution in mix-up augmentations. Default is 1.0.
        min_box_scale (float): Minimum relative size (height and width) for a bounding box to be considered valid during target encoding. 
                               This is with respect to the input image size.
                               For example, if the input image size is (416, 416) and `min_box_scale = 0.01`, 
                               the minimum height and width for valid bounding boxes is 416*0.01 = 4.16 pixels.
                               Default is 0.01.
        display_name (optional, str): Display name used to represent the dataset when printed by `__repr__`.

    Note: while `scale_anchors` is a required parameter, 
    they can be optionally regenerated using the method `regenerate_scale_anchors(...)`.
    This will generate a new set of anchors using the dataset and k-means clustering.
    '''
    def __init__(self, 
                 root: str,
                 label_path: str,
                 scale_anchors: List[torch.Tensor],
                 strides: List[Union[int, Tuple[int, int]]],
                 ignore_threshold: float = 0.5,
                 default_input_size: Union[int, Tuple[int, int]] = (416, 416),
                 single_augs: Optional[Callable] = None,
                 multi_augs: Union[Literal['mosaic', 'mixup'], List[Literal['mosaic', 'mixup']]] = 'mosaic',
                 post_multi_augs: Optional[Callable] = None,
                 multi_aug_prob: float = 0.0,
                 mixup_alpha: float = 1.0,
                 min_box_scale: float = 0.01,
                 display_name: Optional[str] = None):
        super().__init__()

        self.root = root
        self.label_path = label_path
        self.scale_anchors = scale_anchors
        self.strides = [misc.make_tuple(stride) for stride in strides]
        self.default_input_size = misc.make_tuple(default_input_size)
        self.ignore_threshold = ignore_threshold
        self.single_augs = single_augs
        self.multi_augs = multi_augs
        self.post_multi_augs = post_multi_augs
        self.multi_aug_prob = multi_aug_prob
        self.mixup_alpha = mixup_alpha
        self.min_box_scale = min_box_scale
        self.display_name = '' if display_name is None else display_name
        
        self.mixup_beta_dist = torch.distributions.Beta(mixup_alpha, mixup_alpha)
        self.multi_aug_map = {
            'mosaic': self.load_mosaic_image_and_targets,
            'mixup': self.load_mixup_image_and_targets
        }

        self.anchors_info = self._get_anchors_info() # Dictionary with anchor/scale information for encoding
        self.class_names, self.class_clrs, self.class_to_idx = load_classes(label_path, clr_shuffle_seed = 42)
             
    def __repr__(self) -> str:
        '''
        Returns a readable string representation of the dataset.
        Includes dataset name, number of samples, number of classes,
        strides, default size, and example image and target shapes.
        '''
        examp_img, examp_targs = self.__getitem__(0)
        if isinstance(examp_img, torch.Tensor):
            img_shape = tuple(examp_img.shape)
        else:
            img_shape = 'N/A'

        targ_shapes = [tuple(targ.shape) for targ in examp_targs]

        dataset_str = f'''\
        {BOLD_START}Dataset:{BOLD_END} {self.display_name}
            {BOLD_START}Root location:{BOLD_END} {self.root}
            {BOLD_START}Number of samples:{BOLD_END} {self.__len__()}
            {BOLD_START}Number of classes:{BOLD_END} {len(self.class_names)}
            {BOLD_START}Strides:{BOLD_END} {self.strides}
            {BOLD_START}Default Input Size:{BOLD_END} {self.default_input_size}
            {BOLD_START}Default Image shape:{BOLD_END} {img_shape}
            {BOLD_START}Default Target shapes:{BOLD_END} {targ_shapes}
        '''
        return textwrap.dedent(dataset_str)
    
    def __getitem__(
            self, 
            input_info: Union[Tuple[int, Union[int, Tuple[int, int]]], int]
    ) -> Tuple[Union[Image.Image, torch.Tensor], List[torch.Tensor]]:
        '''
        Gets the transformed image and encoded targets for a given index.

        Args:
            input_info (Tuple[int, Union[int, Tuple[int, int]]] or int): 
                A single image index or a tuple containing the both image index and an input size of form (height, width).
                If only the image index is given, the returned image is resized to `default_image_size`.
                If a tuple in the form `(idx, input_size)` is given, the returned image is resized to `input_size`.

        Returns:
            img (Image.Image or torch.Tensor): The transformed image based on `idx`. 
                                               If `self.multi_aug_prob > 0`, there is a `self.multi_aug_prob * 100%` chance 
                                               that the image undergoes a multi-image augmentation (e.g. mosaic or mixup).
            scale_targs (List[torch.Tensor]): List of YOLOv3-encoded target tensors, one per scale of the model.
                                              Each has shape: (num_anchors, fmap_h, fmap_w, 5 + C),
                                              where the last dimension represents 
                                              (xmin, xmax, width, height, object confidence, class scores).
        '''
        
        if isinstance(input_info, tuple):
            idx, input_size = input_info

        else:
            idx, input_size = input_info, self.default_input_size

        # P(X < x) = x for X ~ Unif[0, 1]
        if random.uniform(0, 1) < self.multi_aug_prob:
            if isinstance(self.multi_augs, list):
                load_multi_img_targs = self.multi_aug_map[random.choice(self.multi_augs)]
            else:
                load_multi_img_targs = self.multi_aug_map[self.multi_augs]

            return load_multi_img_targs(idx, input_size)
        else:
            return self.load_single_image_and_targets(idx, input_size) 
        
    def load_single_image_and_targets(
            self, 
            idx: int,
            input_size: Union[int, Tuple[int, int]],
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        '''
        Loads a single transformed image and its encoded targets, given an index.

        Args:
            idx (int): Image index.
            input_size (int or Tuple[int, int]): The size (height, width) to resize the image into. 
                                                 If `int`, it is assumed to be square.

        Returns:
            img (torch.Tensor): The transformed image at `idx`.
                                If provided, data augmentations from `single_augs` are first applied.
                                The image is then resized (letterbox) and converted 
                                to a tensor with normalized pixels in [0, 1].
            scale_targs (List[torch.Tensor]): List of YOLOv3-encoded target tensors, one per scale of the model.
                                              Each has shape: (num_anchors, fmap_h, fmap_w, 5 + C),
                                              where the last dimension represents 
                                              (xmin, xmax, width, height, object confidence, class scores)
        '''
        input_size = misc.make_tuple(input_size)

        img = self.get_img(idx)
        anno_info = self.get_anno_info(idx)

        # Data augmentations
        if self.single_augs is not None:
            img, anno_info = self.single_augs(img, anno_info)

        # Converting to tensor and resizing with letter box
        img = F.to_image(img)
        img, anno_info = transforms.functional_letterbox(
            img = img, anno_info = anno_info, 
            size = input_size, fill = 114, return_bbox_fmt = 'xyxy'
        )
        img = F.to_dtype(img, dtype = torch.float32, scale = True) # Rescales to [0, 1]
        
        # Encoding targets
        scale_targs = self._encode_yolov3_targets(anno_info = anno_info)

        return img, scale_targs
    
    def load_mosaic_image_and_targets(
            self, 
            idx: int,
            input_size: Union[int, Tuple[int, int]]
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        '''
        Loads a 2x2 mosaic of 4 images along with their corresponding encoded targets. 
        The top-left image panel is given by `idx`, while the rest of the 3 images are randomlt chosen from the dataset.
        
        Reference: https://gmongaras.medium.com/yolox-explanation-mosaic-and-mixup-for-data-augmentation-3839465a3adf

        Args:
            idx (int): Image index for the top-left image panel of the mosaic.
            input_size (int or Tuple[int, int]): The size to resize the mosaic into. 
                                                 All mosaic image panels are also resized based on `input_size` 
                                                 (aspect ratio preserved) before pasting onto the mosaic's pre-cropped canvas.
                                                 If `tuple`, it should have the form (height, width).
                                                 If `int`, the reshape size is assumed to be square.
        Returns:
            mosaic_img (torch.Tensor): The cropped and transformed mosaic image. 
                                       If provided, data augmentations from `post_multi_augs` are applied post-crop.
                                       The image is then converted to a tensor with normalized pixels in [0, 1].
            scale_targs (List[torch.Tensor]): List of YOLOv3-encoded target tensors, one per scale of the model.
                                              Each has shape: (num_anchors, fmap_h, fmap_w, 5 + C),
                                              where the last dimension represents 
                                              (xmin, xmax, width, height, object confidence, class scores). 
        '''
        input_size = misc.make_tuple(input_size)
        input_h, input_w = input_size

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
            
            # Resize image and bounding boxes (aspect ratio preserved)
            img, anno_info = transforms.functional_resize(
                img = img, anno_info = anno_info, 
                size = (scaled_h, scaled_w), return_bbox_fmt = 'xyxy'
            )
            
            # Paste image such that one of its inner corner aligns with canvas center
            paste_x, paste_y = img_pos

            if paste_x == 0:
                paste_x += input_w - scaled_w

            if paste_y == 0:
                paste_y += input_h - scaled_h
                
            mosaic_img.paste(img, (paste_x, paste_y))

            # Adjust bounding box positions
            bboxes = anno_info['boxes']
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
            # within a (input_w/2)x(input_h/2) box centered on the mosaic image
            # This ensures that the final cropped image will always contain a part of all 4 images
        crop_cx = random.randint(3 * input_w // 4, 5 * input_w // 4)
        crop_cy = random.randint(3 * input_h // 4, 5 * input_h // 4)

        # Crop region
        crop_xmin = crop_cx - input_w // 2
        crop_ymin = crop_cy - input_h // 2
        crop_xmax = crop_xmin + input_w
        crop_ymax = crop_ymin + input_h

        mosaic_img = mosaic_img.crop((crop_xmin, crop_ymin, crop_xmax, crop_ymax))

        # Shift and clamp bboxes to image boundaries
        all_bboxes[:, ::2] = (all_bboxes[:, ::2] - crop_xmin).clamp(0, input_w)
        all_bboxes[:, 1::2] = (all_bboxes[:, 1::2] - crop_ymin).clamp(0, input_h)

        # Remove bboxes that are outside of boundaries (i.e. those with almost no width/height post-clamp)
        valid_mask = ((all_bboxes[:, 2] - all_bboxes[:, 0] > 5) & 
                      (all_bboxes[:, 3] - all_bboxes[:, 1] > 5))
        
        # Create mosaic annotation info in a format suitable for v2 transforms
        mosaic_anno_info = {
            'boxes': BoundingBoxes(
                data = all_bboxes[valid_mask],
                canvas_size = input_size,
                format = 'XYXY'
            ),
            'labels': all_labels[valid_mask]
        }

        if self.post_multi_augs is not None:
            mosaic_img, mosaic_anno_info = self.post_multi_augs(mosaic_img, mosaic_anno_info)

        mosaic_img = F.to_image(mosaic_img)
        mosaic_img = F.to_dtype(mosaic_img, dtype = torch.float32, scale = True) # Rescales to [0, 1]
        scale_targs = self._encode_yolov3_targets(anno_info = mosaic_anno_info)

        return mosaic_img, scale_targs
    
    def load_mixup_image_and_targets(
            self, 
            idx: int,
            input_size: Union[int, Tuple[int, int]]
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        '''
        Loads a mix-up of 2 image along with their corresponding encoded targets. 
        One of the images in the mix-up is given by `idx`, while the other is randomly chosen from the dataset.

        Reference: https://gmongaras.medium.com/yolox-explanation-mosaic-and-mixup-for-data-augmentation-3839465a3adf

        Args:
            idx (int): Image index for one of the images used in the mix-up
            input_size (int or Tuple[int, int]): The size of the mix-up image.
                                                 All images in the mix-up are resized to `input_size` 
                                                 using a letterbox transform.
                                                 If `tuple`, it should have the form (height, width).
                                                 If `int`, the reshape size is assumed to be square.
        Returns:
            mixup_img (torch.Tensor): The transformed mix-up image.
                                      If provided, data augmentations from `post_multi_augs` are applied after the mix-up.
                                       The image is then converted to a tensor with normalized pixels in [0, 1].
            scale_targs (List[torch.Tensor]): List of encoded target tensors, one per scale of the model.
                                              Each has shape: (num_anchors, fmap_h, fmap_w, 5 + C),
                                              where the last dimension represents 
                                              (xmin, xmax, width, height, object confidence, class scores).
                                              Note that the object confidence values for positive cells are 
                                              scaled by the mixup lambda (or 1 - lambda) to represent a softer P(object).
        '''
        lam = self.mixup_beta_dist.sample()
        img_weights = [lam, 1 - lam]
        input_size = misc.make_tuple(input_size)

        all_idxs = [idx] + random.sample(range(self.__len__()), 1)
        all_imgs = [] # Stores image tensors that have been rescaled to [0, 1]
        all_anno_infos = [] # Stores corresponding annotation information for image tensors

        for img_idx in all_idxs:
            img = self.get_img(img_idx)
            anno_info = self.get_anno_info(img_idx)
            
            # Resize image to a common input shape, while preserving aspect ratio
            img, anno_info = transforms.functional_letterbox(img = img, anno_info = anno_info, 
                                                             size = input_size, fill = 114)
            img = F.to_image(img)
            img = F.to_dtype(img, dtype = torch.float32, scale = True)
            
            all_imgs.append(img)
            all_anno_infos.append(anno_info)
            
        mixup_img = img_weights[0] * all_imgs[0] + img_weights[1] * all_imgs[1]

        mixup_labels = torch.concat([anno_info['labels'] for anno_info in all_anno_infos])
        mixup_bboxes = torch.concat([anno_info['boxes'] for anno_info in all_anno_infos], dim = 0)
        mixup_weights = torch.concat([
            weight * torch.ones_like(anno_info['labels'])
            for weight, anno_info in zip(img_weights, all_anno_infos)
        ])

        mixup_anno_info = {
            'labels': mixup_labels,
            'boxes': BoundingBoxes(
                data = mixup_bboxes,
                format = all_anno_infos[0]['boxes'].format,
                canvas_size = input_size
            )
        }

        if self.post_multi_augs is not None:
            mixup_img, mixup_anno_info = self.post_multi_augs(mixup_img, mixup_anno_info)
        scale_targs = self._encode_yolov3_targets(anno_info = mixup_anno_info, obj_weights = mixup_weights)

        return mixup_img, scale_targs

    def regenerate_scale_anchors(self, 
                                 anchors_per_scale: Tuple[int] = (3, 3, 3), 
                                 max_iters: int = 1000,
                                 update_method: Literal['mean', 'median'] = 'mean',
                                 input_size: Optional[Tuple[int, int]] = None):
        '''
        Generates anchor boxes for each detection scale using k-means clustering 
        over the dataset's bounding box widths and heights.

        The resulting anchors are stored in `self.scale_anchors`, with additional information stored in `self.anchors_info`.
        Moreover, the bounding box (width, height) pairs are stored in `self.bbox_whs` if not already defined.

        Args:
            anchors_per_scale (Tuple[int]): Number of anchors to assign to each detection scale.
                                            The order of the scales are assumed to be from largest to smallest.
                                            The total number of anchors (k) is the sum of this tuple.
                                            Default is (3, 3, 3) for three scales with three anchors each (YOLOv3 default).
            max_iters (int): Maximum number of iterations to run k-means clustering. Default is 1000.
            update_method (Literal['mean', 'median']): Method to update the cluster centroids.
                                                        - mean: Use the mean of the grouped bboxes.
                                                        - median: Use the median of the grouped bboxes.
            input_size (Optional[Tuple[int, int]]): Input size in (height, width) format.
                                                    If not provided, defaults to `self.default_input_size`.
        '''
        input_size = self.default_input_size if input_size is None else input_size
        num_anchors = sum(anchors_per_scale)

        # Get all bounding box (width, height) pairs if not available
        if not hasattr(self, 'bbox_whs'):
            bbox_whs = []
            for img_idx in range(self.__len__()):
                anno_info = self.get_anno_info(img_idx)
                img_h, img_w = anno_info['boxes'].canvas_size # (height, width)
                bboxes = box_convert(
                    boxes = anno_info['boxes'], 
                    in_fmt = anno_info['boxes'].format.value.lower(),
                    out_fmt = 'xyxy'
                )
                
                whs = bboxes[:, 2:4] - bboxes[:, :2]
                whs[:, 0] /= img_w
                whs[:, 1] /= img_h

                bbox_whs.append(whs)
            self.bbox_whs = torch.concat(bbox_whs, dim = 0)

        # Perform k-means clustering on the bbox_whs set
        # Note: the returned anchors are ordered in ascending order of area
        all_anchors = wh_k_means(bbox_whs = self.bbox_whs,
                                 k = num_anchors,
                                 max_iter = max_iters,
                                 update_method = update_method,
                                 round_whs = True,
                                 area_order = 'desc',
                                 input_size = input_size)
        
        # Scale order: large -> small
        self.scale_anchors = list(torch.split(all_anchors, anchors_per_scale, dim = 0))
        self.anchors_info = self._get_anchors_info() # Update anchor information

    def _get_anchors_info(self) -> Dict[str, Any]:
        '''
        Gets anchor and scale information.

        Returns:
            anchors_info (Dict[str, Any]): Dictionary of anchor and scale information.
                The keys are:
                    - anchors_wh (torch.Tensor): Tensor of all (width, height) pairs for the anchors.
                                                    Shape is (num_tot_anchors, 2).
                    - num_tot_anchors (int): Total number of anchors, summed across all detection scales.
                    - anchor_scale_idxs (torch.Tensor): Tensor indicating which detection scale each anchor corresponds to.
                                                        Shape is (num_tot_anchors,).
                    - anchors_per_scale (torch.Tensor): Tensor indicating number of anchors per detection scale.
                                                        Shape is (num_scales, ).
        '''
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

    def _encode_yolov3_targets(
            self, 
            anno_info: Dict[str, Any], 
            obj_weights: Optional[torch.Tensor] = None
        ) -> List[torch.Tensor]:
        '''
        Creates a list of YOLOv3-encoded targets (one per detection scale), given annotation information.

        Args:
            anno_info (Dict[str, Any]): Dictionary of annotation information to construct YOLOv3-encoded targets from.
                                        It should contain the following keys:
                                            - labels (torch.Tensor): Tensor of label indices with shape (num_objects,).
                                            - boxes (BoundingBoxes): BoundingBox object storing bounding box coordinates
                                                                     in XYXY format and in pixel units (canvas is the image size). 
                                                                     Shape is (num_objects, 4).
            obj_weights (optional, torch.Tensor): Object confidences (weights) for each object in `labels`. 
                                                  Shape should be (num_objects,). 
                                                  If not provided, it is assumed that each object has 
                                                  an object confidence of P(object) = 1.
        Returns:
            scale_targs: (List[torch.Tensor]): List of YOLOv3-encoded ground truth targets, one per detection scale.
                                               Each element is a tensor of shape: (batch_size, num_anchors, fmap_h, fmap_w, 5 + C),
                                               where the last dimension represents 
                                               (xmin, ymin, width, height, object confidence, class scores).
        '''
        anchors_wh = self.anchors_info['anchors_wh']
        num_tot_anchors = self.anchors_info['num_tot_anchors']
        anchor_scale_idxs = self.anchors_info['anchor_scale_idxs']
        anchors_per_scale = self.anchors_info['anchors_per_scale']
        
        input_size = anno_info['boxes'].canvas_size
        fmap_sizes = [(input_size[0]//s[0], input_size[1]//s[1]) for s in self.strides]

        # Create placeholder target tensors in the same shape as YOLOv3 logits (one per scale)
        scale_targs = []
        for size, anchors in zip(fmap_sizes, self.scale_anchors):
            scale_targs.append(
                torch.zeros(anchors.shape[0], size[0], size[1], 5 + len(self.class_names))
            )

        labels = anno_info['labels']

        # Shape of bboxes_cxcywh and bboxes_xyxy: (num_bboxes, 4)
        # bboxes_cxcywh and bboxes_xyxy should be in units of the input size (pixel)
        bboxes_xyxy = box_convert(
            boxes = anno_info['boxes'], 
            in_fmt = anno_info['boxes'].format.value.lower(),
            out_fmt = 'xyxy'
        )

        # # Clamp to ensure all bboxes are within image 
        # bboxes_xyxy[:, ::2] = bboxes_xyxy[:, ::2].clamp(0, img_w)
        # bboxes_xyxy[:, 1::2] = bboxes_xyxy[:, 1::2].clamp(0, img_h)
        bboxes_cxcywh = convert.xyxy_to_cxcywh(bboxes_xyxy)

        # Filter out objects that may have been too cut off due to transforms
        min_size_tensor = self.min_box_scale * torch.tensor(input_size).flip(dims = [0]) # (min_width, min_height)
        valid_mask = (bboxes_cxcywh[:, 2:] > min_size_tensor).all(dim = 1)

        bboxes_cxcywh = bboxes_cxcywh[valid_mask]
        bboxes_xyxy = bboxes_xyxy[valid_mask]
        labels = labels[valid_mask]

        if obj_weights is None:
            obj_weights = torch.ones_like(labels) # Each object has P(object) = 1
        else:
            obj_weights = obj_weights[valid_mask]

        # Loop over objects in the image and assign them to an anchor/scale
        for label, bbox_cxcywh, bbox_xyxy, obj_weight in zip(labels, bboxes_cxcywh, bboxes_xyxy, obj_weights):
            cx, cy, w, h = bbox_cxcywh

            cxcy = bbox_cxcywh[:2].repeat(num_tot_anchors, 1) # Shape: (num_tot_anchors, 2)
            bbox_anchors = torch.concat([cxcy, anchors_wh], dim = -1) # Shape: (num_tot_anchors, 4); Format: CXCYWH
            bbox_anchors = convert.cxcywh_to_xyxy(bbox_anchors) # Convert CXCYWH to XYXY

            # -------------------------------
            # Assign Responsible Anchor Box
            # -------------------------------
            ious = evaluate.calc_ious(bbox_xyxy.unsqueeze(0), bbox_anchors).squeeze() # Shape: (num_tot_anchors,)
            _, topk_idxs = ious.topk(k = 4)  # Only consider top 4 anchors for assignment
            for topk_idx in topk_idxs:
                scale_idx = anchor_scale_idxs[topk_idx]
                anchor_idx = topk_idx - anchors_per_scale[:scale_idx].sum()

                fmap_h, fmap_w = fmap_sizes[scale_idx] # Height, width of the fmap
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

                if scale_targs[scale_idx][anchor_idx, grid_y, grid_x, 4] == 0:
                    targ_bbox = torch.tensor([targ_x, targ_y, targ_w, targ_h, obj_weight], dtype = torch.float32)
                    scale_targs[scale_idx][anchor_idx, grid_y, grid_x, :5] = targ_bbox

                    # One-hot encode the class label
                    # Note: I don't think I need to multiply by obj_weight, since this is P(class|object)
                    scale_targs[scale_idx][anchor_idx, grid_y, grid_x, 5 + label] = 1

                    # -------------------------------
                    # Assign Ignore Anchor Boxes
                    # -------------------------------
                    ignore_mask = (ious >= self.ignore_threshold)
                    ignore_mask[topk_idx] = False
                    ignore_idxs = torch.where(ignore_mask)[0]

                    for ig_idx in ignore_idxs:
                        ig_scale_idx = anchor_scale_idxs[ig_idx]
                        ig_anchor_idx = ig_idx - anchors_per_scale[:ig_scale_idx].sum()
                        
                        ig_fmap_h, ig_fmap_w = fmap_sizes[ig_scale_idx]
                        ig_stride_h, ig_stride_w = self.strides[ig_scale_idx]

                        ig_grid_x = min(int(cx / ig_stride_w), ig_fmap_w - 1)
                        ig_grid_y = min(int(cy / ig_stride_h), ig_fmap_h - 1)
                                        
                        # Object score of -1 implies ignore this cell during loss calculation
                        if scale_targs[ig_scale_idx][ig_anchor_idx, ig_grid_y, ig_grid_x, 4] == 0:
                            scale_targs[ig_scale_idx][ig_anchor_idx, ig_grid_y, ig_grid_x, 4] = -1

                    break  # Once assigned, stop trying to assign
        return scale_targs
    
    @abstractmethod
    def __len__(self) -> int:
        '''
        Must be implemented by subclasses.

        Returns:
            int: The number of images in the dataset.
        '''
        pass

    @abstractmethod
    def get_img(self, idx: int) -> Image.Image:
        '''
        Loads an original image from the dataset and converts it to RGB format

        Args:
            idx (int): Index of the image in the dataset.

        Returns:
            Image.Image: The original image in RGB format,
                          before any additional transforms are applied.
        '''
        pass

    @abstractmethod
    def get_anno_info(self, idx: int) -> Dict[str, Any]:
        '''
        Loads the annotation information (labels and bounding boxes)
        for an original image, untransformed image in the dataset.

        Args:
            idx (int): Index of the image in the dataset.

        Returns:
            Dict[str, Any]: Annotation dictionary for the original image at index `idx`,
                            before any additional transforms are applied. It consists of the keys:
                                - labels (torch.Tensor): Tensor of label indices for each object. 
                                                         Shape is (num_objects,).
                                - boxes (BoundingBoxes): BoundingBox object storing bounding box coordinates
                                                         in XYXY format and in pixel units 
                                                         (canvas_size is the image size). 
                                                         Shape is (num_objects, 4).
        '''
        pass