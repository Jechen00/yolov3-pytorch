#####################################
# Imports & Dependencies
#####################################
import torch
from torchvision.tv_tensors import BoundingBoxes
from torchvision.datasets.utils import download_and_extract_archive

import os
import random
from PIL import Image
import xml.etree.ElementTree as ET
from typing import Optional, List, Union, Callable, Tuple, Literal

from src.utils.constants import BOLD_START, BOLD_END
from src.data_setup.dataset_utils import DetectionDatasetBase


# From PyTorch docs: 
    # https://docs.pytorch.org/vision/main/_modules/torchvision/datasets/voc.html#VOCDetection
VOC_DATA_URLS = {
    'trainval2012': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar',
        'filename': 'VOCtrainval_11-May-2012.tar',
        'md5': '6cd6e144f989b92b3379bac3b3de84fd',
        'orig_dir': 'VOCdevkit/VOC2012',
        'base_dir': 'VOCdevkit/VOC2012_trainval' 
    },
    'trainval2007': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar',
        'filename': 'VOCtrainval_06-Nov-2007.tar',
        'md5': 'c52e279531787c972589f7e41ab4ae64',
        'orig_dir': 'VOCdevkit/VOC2007', 
        'base_dir': 'VOCdevkit/VOC2007_trainval',
    },
    'test2007': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar',
        'filename': 'VOCtest_06-Nov-2007.tar',
        'md5': 'b6e924de25625d8de591ea690078ad9f',
        'orig_dir': 'VOCdevkit/VOC2007', 
        'base_dir': 'VOCdevkit/VOC2007_test'
    }
}


#####################################
# VOC Dataset Class
#####################################
class VOCDataset(DetectionDatasetBase):
    '''
    Pascal VOC training and validation/test dataset.

    The training set combines the trainval data from  Pascal VOC 2007 and 2012. 
    The validation/test set is the test data from Pascal VOC 2007.

    Args:
        root (str): The directory to the VOCdevit folder containing the 
                    relevant 2007 and 2012 Pascal VOC trainval and/or test data. 
                    See `VOC_DATA_URLS` for the specific folder names.
                    If the specified folders does not exist in `root`, the dataset will be downloaded.
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
        split (Literal['train', 'val']): Whether to construct the training dataset ('train') or the validation dataset ('val').
                                         Default is 'train'.
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
                 split: Literal['train', 'val'] = 'train', 
                 ignore_threshold: float = 0.5,
                 single_augs: Optional[Callable] = None,
                 multi_augs: Union[Literal['mosaic', 'mixup'], List[Literal['mosaic', 'mixup']]] = 'mosaic',
                 post_multi_augs: Optional[Callable] = None,
                 multi_aug_prob: float = 0.0,
                 mixup_alpha: float = 0.5,
                 min_box_scale: float = 0.01,
                 max_imgs: Optional[int] = None):
        assert split in ['train', 'val'], (
            "Invalid dataset split: expected 'train' or 'val'. "
            "If you're using this dataset for testing purposes, consider setting `split='val'`."
        )
        if max_imgs is not None:
            assert max_imgs > 0, ('Must have `max_imgs > 0` or `max_imgs = None`')

        self.split = split
        self.max_imgs = max_imgs
        
        if split == 'train':
            data_keys = ['trainval2007', 'trainval2012']
            id_txt = 'trainval.txt'
            display_name = 'Pascal VOC 2012+2007'
        elif split == 'val':
            data_keys = ['test2007']
            id_txt = 'test.txt'
            display_name = 'Pascal VOC 2007 Test'
        label_path = os.path.join(root, 'voc.names') # Path to VOC class names

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
        
        self.voc_paths, self.img_paths, self.anno_paths = [], [], []
        for key in data_keys:
            url_info = VOC_DATA_URLS[key]

            paths = {'data_key': key}
            paths['dataset'] = os.path.join(root, url_info['base_dir']) # file path to VOC dataset directory
            paths['annotations'] = os.path.join(paths['dataset'], 'Annotations') # file path to .xml annotations
            paths['imgs'] = os.path.join(paths['dataset'], 'JPEGImages') # file path to .jpg images
            
            # Check if VOC dataset directory exists, if not download it
            if not os.path.exists(paths['dataset']):
                print(f"{BOLD_START}[DOWNLOADING]{BOLD_END} Dataset VOC{key} to {paths['dataset']}")
                download_and_extract_archive(url = url_info['url'],
                                             download_root = root, 
                                             filename = url_info['filename'],
                                             md5 = url_info['md5'],
                                             remove_finished = True)
                
                # Rename the directory containing the dataset after extraction
                os.rename(os.path.join(root, url_info['orig_dir']), paths['dataset'])
                
            # Get image for the dataset
            imgs, annos = [], []
            id_path = os.path.join(paths['dataset'], 'ImageSets', 'Main', id_txt)
            with open(id_path, 'r') as f:
                for img_id in f.readlines():
                    annos.append(os.path.join(paths['annotations'], f'{img_id.strip()}.xml'))
                    imgs.append(os.path.join(paths['imgs'], f'{img_id.strip()}.jpg'))
                
            self.voc_paths.append(paths)
            self.img_paths += imgs
            self.anno_paths += annos
        
        # Optionally reduce dataset size
        if (max_imgs is not None) and (max_imgs < self.__len__()):
            samp_idxs = random.sample(range(self.__len__()), max_imgs)

            self.img_paths = [self.img_paths[i] for i in samp_idxs]
            self.anno_paths = [self.anno_paths[i] for i in samp_idxs]

    def __len__(self) -> int:
        '''
        Returns the number of images in the dataset.
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
        return Image.open(self.img_paths[idx]).convert('RGB')
    
    def get_anno_info(self, idx: int) -> dict:
        '''
        Parses the annotation XML file for a given index to retrieve 
        annotation information (labels and bounding boxes) for 
        the original, untransformed image in the dataset.

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
        xml_root = ET.parse(self.anno_paths[idx]).getroot()
        size = xml_root.find('size')
        canvas_size = (int(size.find('height').text),  int(size.find('width').text))

        labels, bboxes = [], []
        for obj in xml_root.findall('object'):
            labels.append(self.class_to_idx[obj.find('name').text])
            
            bnd_box = obj.find('bndbox')
            bboxes.append(
                [float(bnd_box.find('xmin').text), 
                 float(bnd_box.find('ymin').text), 
                 float(bnd_box.find('xmax').text), 
                 float(bnd_box.find('ymax').text)]
            )
        
        if len(bboxes) > 0:
            # Every image in VOC should have an object
            anno_info = {
                'labels': torch.tensor(labels, dtype = torch.long),
                'boxes': BoundingBoxes(
                    data = torch.tensor(bboxes, dtype = torch.float32), 
                    format = 'XYXY', 
                    canvas_size = canvas_size)
            }
        else:
            anno_info = {
                    'labels': torch.empty(0, dtype = torch.long),
                    'boxes': BoundingBoxes(data = torch.empty((0, 4)), format = 'XYXY', 
                                           canvas_size = canvas_size)
            }

        return anno_info