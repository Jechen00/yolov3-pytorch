#####################################
# Imports & Dependencies
#####################################
import torch
from torch.utils.data import Dataset
from torchvision.tv_tensors import BoundingBoxes

import os
import json
import random
from PIL import Image
from collections import defaultdict
from typing import Optional, List, Union, Callable, Tuple

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
class COCODataset(Dataset, DetectionDatasetBase):
    '''
    input_size (int or Tuple[int, int]): Input image size, used to compute feature map sizes.
                                            If `resize=True`, all images will be resized to this.
                                            If `resize=False`, the user-supplied `transforms` 
                                            must resize to `input_size`, or the target encoding may be misaligned.
    scale_anchors (List[torch.tensor]): List of anchor tensors for each output scale of the model.
                                        Each element has shape: (num_anchors, 2), where the last dimension gives 
                                        the (width, height) of the anchor in units of the input size (pixels).
    strides (List[Union[int, Tuple[int, int]]]): List of strides corresponding to each scale in `scale_anchors`. 
                                                    Each stride represents the downsampling factor between the input image 
                                                    and the feature map at that scale.
    resize (bool): Whether to apply image resizing to `input_size`, pixel rescaling to [0, 1], 
                    and tensor conversion *after* any user-supplied `transforms`. 
                    If `resize = True`, user-supplied `transforms` should exclude 
                    these transformations to avoid duplication. Default is True.
    ignore_threshold (float): The IoU threshold used to encode which anchor boxes will be ignored during loss calculation.
                                Anchor boxes that do not have the highest IoU with a ground truth box, 
                                but have an IoU >= ignore_threshold, 
                                will be marked with an encoded index of `-1` to indicate they should be ignored.
                                Default value is 0.5.
    '''
    def __init__(self, 
                 root: str, 
                 scale_anchors: List[torch.Tensor],
                 input_size: Union[int, Tuple[int, int]],
                 strides: List[Union[int, Tuple[int, int]]],
                 train: bool = True, 
                 transforms: Optional[Callable] = None,
                 resize: bool = True,
                 ignore_threshold: float = 0.5,
                 max_imgs: Optional[int] = None):
        Dataset.__init__(self)
        self.train = train
        self.transforms = transforms
        self.resize = resize
        self.max_imgs = max_imgs
        
        if train:
            data_keys = ['train2014', 'val2014', 'anno2014']
            part_file = os.path.join(root, 'trainvalno5k.part') # Path to partition file
            dataset_name = 'MS-COCO 2014 TrainVal35K'
        else:
            data_keys = ['val2014', 'anno2014']
            part_file = os.path.join(root, '5k.part') # Path to partition file
            dataset_name = 'MS-COCO 2014 Val5K'
        label_path = os.path.join(root, 'coco.names') # Path to COCO class labels

        # Initialize base detection dataset attributes 
            #  (scale_anchors, fmap_sizes, classes, resize_transforms, etc.)
        DetectionDatasetBase.__init__(self, root = root, scale_anchors = scale_anchors,
                                      input_size = input_size, strides = strides, 
                                      label_path = label_path, dataset_name = dataset_name,
                                      ignore_threshold = ignore_threshold)

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
    
    def __getitem__(
            self, 
            idx: int
    ) -> Tuple[Union[Image.Image, torch.Tensor], List[torch.Tensor]]:
        '''
        Gets the transformed image and targets for a given index.
        Returns:
            img (Image.Image or torch.Tensor): The transformed image at `idx`.
                                               The exact type of `img` depends 
                                               on the transforms of the dataset.
            scale_targs (List[torch.Tensor]): List of encoded target tensors, one per scale of the model.
                                              Each has shape: (num_anchors, fmap_h, fmap_w, 5 + C)
        '''
        img_file = self.img_paths[idx]
        filename = os.path.basename(img_file)
        
        img = Image.open(img_file).convert('RGB')
        anno_info = self.filename_to_annos[filename]

        if self.transforms is not None:
            img, anno_info = self.transforms(img, anno_info)

        if self.resize:
            img, anno_info = self.resize_transforms(img, anno_info)
        
        scale_targs = self._encode_yolov3_targets(anno_info)
        return img, scale_targs

    def get_img(self, idx: int) -> Image.Image:
        '''
        Loads an image from the dataset (pre-transform).
        '''
        img_file = self.img_paths[idx]
        return Image.open(img_file).convert('RGB')
    
    def get_anno_info(self, idx: int) -> dict:
        '''
        Loads annotation information (labels and bounding boxes)
        for an image from the dataset (pre-transform).
        '''
        img_file = self.img_paths[idx]
        filename = os.path.basename(img_file)
        return  self.filename_to_annos[filename]
        
    def _load_annotations(self, data_keys: List[str]):
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
                bbox_tensor = convert.center_to_corner_format(bbox_tensor) # Converts CXCYWH -> XYXY

                # Change value of filename_to_annos to a suiable format for v2 transforms
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