#####################################
# Imports & Dependencies
#####################################
import torch
from torchvision.tv_tensors import BoundingBoxes

import os
import json
import random
from PIL import Image
from collections import defaultdict
from typing import Optional, List, Union, Callable, Tuple, Literal, Dict, Any

from src.utils import convert
from src.data_setup import dataset_utils
from src.data_setup.dataset_utils import DetectionDatasetBase


COCO_2014_URLS = {
    'train2014': {
        'url': 'http://images.cocodataset.org/zips/train2014.zip',
        'base_dir': 'images/train2014'
    },
    'val2014': {
        'url': 'http://images.cocodataset.org/zips/val2014.zip',
        'base_dir': 'images/val2014'
    },
    'anno2014': {
        'url': 'http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
        'base_dir': 'annotations'
    }
}

    
#####################################
# COCO Dataset Class
#####################################
class COCODataset(DetectionDatasetBase):
    '''
    MS COCO training and validation/testing dataset.

    The training set consists of all ~82.8k images from the MS COCO 2014 training set (train2014), 
    along with ~35.5k images from the MS COCO 2014 validation set (val2014).
    The validation/testing set consists of the remaining ~5k images from val2014.

    Args:
        root (str): The directory to download the dataset to, if needed.
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
        train (bool): Whether to construct the training dataset or the validation/testing dataset.
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
        max_imgs (optional, int): The maximum number of images to include in the dataset. 
                                  If provided, the images are randomly sampled from the full dataset.
                                  If not provided, all available images are included.
    '''
    def __init__(self, 
                 root: str, 
                 scale_anchors: List[torch.Tensor],
                 strides: List[Union[int, Tuple[int, int]]],
                 default_input_size: Union[int, Tuple[int, int]] = (416, 416),
                 train: bool = True, 
                 ignore_threshold: float = 0.5,
                 single_augs: Optional[Callable] = None,
                 multi_augs: Union[Literal['mosaic', 'mixup'], List[Literal['mosaic', 'mixup']]] = 'mosaic',
                 post_multi_augs: Optional[Callable] = None,
                 multi_aug_prob: float = 0.0,
                 mixup_alpha: float = 0.5,
                 min_box_scale: float = 0.01,
                 max_imgs: Optional[int] = None):
        self.train = train
        self.max_imgs = max_imgs
        
        if train:
            data_keys = ['train2014', 'val2014', 'anno2014']
            part_file = os.path.join(root, 'trainvalno5k.part') # Path to partition file
            display_name = 'MS COCO 2014 TrainVal35K'
        else:
            data_keys = ['val2014', 'anno2014']
            part_file = os.path.join(root, '5k.part') # Path to partition file
            display_name = 'MS COCO 2014 Val5K'
        label_path = os.path.join(root, 'coco.names') # Path to COCO class labels

        # Initialize base detection dataset attributes 
        super().__init__(
            root = root, 
            label_path = label_path, 
            display_name = display_name,
            scale_anchors = scale_anchors,
            strides = strides, 
            default_input_size = default_input_size,
            ignore_threshold = ignore_threshold,
            single_augs = single_augs, 
            multi_augs = multi_augs,
            post_multi_augs = post_multi_augs,
            multi_aug_prob = multi_aug_prob, 
            mixup_alpha = mixup_alpha,
            min_box_scale = min_box_scale
        )

        # Check if COCO dataset directory exists, if not download it
        self.coco_paths = {}
        for key in data_keys:
            url_info = COCO_2014_URLS[key]
            data_path = os.path.join(root, url_info['base_dir'])

            if not os.path.exists(data_path):
                download_dir = os.path.dirname(data_path)
                dataset_utils.download_with_curl(
                    url = url_info['url'], 
                    download_dir = download_dir, 
                    unzip = True
                )

            self.coco_paths[key] = data_path
        
        # Image paths from the partition file
        with open(part_file, 'r') as f:
            self.img_paths = [os.path.join(self.root, line.strip().lstrip('/')) for line in f]
            
        # Maps image filenames to their annotations (dictionaries with labels and bounding boxes)
        self.filename_to_annos = self._load_annotations(data_keys = data_keys[:-1])
        
        # Reduce dataset size
        if max_imgs is not None:
            assert max_imgs > 0, ('Must have `max_imgs > 0` or `max_imgs = None`')
            max_imgs = min(max_imgs, self.__len__())

            samp_idxs = random.sample(range(self.__len__()), max_imgs)
            self.img_paths = [self.img_paths[i] for i in samp_idxs]
            
            selected_filenames = set(os.path.basename(p) for p in self.img_paths)
            self.filename_to_annos = {k: v for k, v in self.filename_to_annos.items() 
                                      if k in selected_filenames}
    
    def __len__(self) -> int:
        '''
        Gives the number of images in the dataset.
        '''
        return len(self.img_paths)

    def get_img(self, idx: int) -> Image.Image:
        '''
        Loads an original image from the dataset and converts it to RGB format

        Args:
            idx (int): Index of the image in the dataset.

        Returns:
            Image.Image: The original image in RGB format,
                          before any additional transforms are applied.
        '''
        img_file = self.img_paths[idx]
        return Image.open(img_file).convert('RGB')
    
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
                                                            (canvas is the image size). 
                                                            Shape is (num_objects, 4).
        '''
        img_file = self.img_paths[idx]
        filename = os.path.basename(img_file)
        return  self.filename_to_annos[filename]
        
    def _load_annotations(self, data_keys: List[Literal['train2014', 'val2014']]) -> Dict[str, Dict[str, Any]]:
        '''
        Loads and formats annotation information from MS COCO 2014 JSON files for the specified subsets.

        Args: 
            data_keys (List[Literal['train2014', 'val2014']]):
                 List of dataset keys specifying which MS COCO 2014 annotation files to load.
                    - training set: annotations are loaded from train2014 (~82.8k images) and val2014 (~35.5k images).
                    - validation/testing set: annotations are loaded from only val2014 (remaining ~5k images)

        Returns:
            filename_to_annos (Dict[str, Dict[str, Any]]): Dictionary mapping image file names (e.g. 'COCO_val2014_000000391895.jpg') 
                                                           to their annotation dictionaries.
                                                           Each annotation dictionary consists of the following keys:
                                                                - labels (torch.Tensor): Tensor of label indices for each object. 
                                                                                         Shape is (num_objects,).
                                                                - boxes (BoundingBoxes): BoundingBox object storing bounding box coordinates
                                                                                         in XYXY format and in pixel units 
                                                                                         (canvas_size is the image size). 
                                                                                         Shape is (num_objects, 4).
        '''
        filename_to_annos = defaultdict(list)
        filename_to_size = {}
        for key in data_keys:
            instance_path = os.path.join(self.coco_paths['anno2014'], f'instances_{key}.json')
            with open(instance_path, 'r') as f:
                coco_json = json.load(f)

            # Maps category ids to class names
            cat_id_to_name = {cat['id']: cat['name'] for cat in coco_json['categories']}

            # Maps image ids to filenames
            img_id_to_filename = {img_info['id']: img_info['file_name'] for img_info in coco_json['images']}
            
            # Maps filenames to their image (height, width)
            filename_to_size.update(
                {img_info['file_name']: (img_info['height'], img_info['width'])
                 for img_info in coco_json['images']}
            )
            
            # Each element in the list, coco_json['annotations'], is a single object of an image
            for anno_info in coco_json['annotations']:      
                filename = img_id_to_filename[anno_info['image_id']]
                class_name = cat_id_to_name[anno_info['category_id']]

                # Each list is (xmin, ymin, w, h, class_idx)
                filename_to_annos[filename].append(
                    anno_info['bbox'] + [self.class_to_idx[class_name]]
                )

        for key, value in filename_to_annos.items():
            if len(value) > 0:
                # Tensor of shape (num_bboxes, 5), where last dim is (xmin, ymin, xmax, ymax, class_idx)
                bbox_tensor = torch.tensor(value, dtype = torch.float32)
                bbox_tensor[:, :2] += bbox_tensor[:, 2:4] / 2 # Converts XYWH -> CXCYWH
                bbox_tensor = convert.cxcywh_to_xyxy(bbox_tensor) # Converts CXCYWH -> XYXY

                # Change value of filename_to_annos to a suitable format for v2 transforms
                filename_to_annos[key] = {
                    'labels': bbox_tensor[:, 4].to(dtype = torch.long),
                    'boxes': BoundingBoxes(data = bbox_tensor[:, :4], format = 'XYXY', 
                                           canvas_size = filename_to_size[key])
                }
            else:
                # This implies image has no annotations/objects
                filename_to_annos[key] = {
                    'labels': torch.empty(0, dtype = torch.long),
                    'boxes': BoundingBoxes(data = torch.empty((0, 4)), format = 'XYXY', 
                                           canvas_size = filename_to_size[key])
                }
        
        return filename_to_annos