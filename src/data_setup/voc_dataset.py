#####################################
# Imports & Dependencies
#####################################
import torch
from torch.utils.data import Dataset
from torchvision.tv_tensors import BoundingBoxes
from torchvision.datasets.utils import download_and_extract_archive

import os
import random
from PIL import Image
import xml.etree.ElementTree as ET
from typing import Optional, List, Union, Callable, Tuple

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
    The training set combines the trainval data from  Pascal VOC 2007 and 2012. 
    The validation/test set is the test data from Pascal VOC 2007.
    
    Dataset based on https://docs.pytorch.org/vision/main/_modules/torchvision/datasets/voc.html#VOCDetection
    '''
    def __init__(self, 
                 root: str, 
                 scale_anchors: List[torch.Tensor],
                 strides: List[Union[int, Tuple[int, int]]],
                 default_input_size: Union[int, Tuple[int, int]],
                 train: bool = True, 
                 single_augs: Optional[Callable] = None,
                 mosaic_augs: Optional[Callable] = None,
                 mosaic_prob: float = 0.0,
                 ignore_threshold: float = 0.5,
                 min_box_scale: float = 0.01,
                 max_imgs: Optional[int] = None):
        self.train = train
        self.max_imgs = max_imgs
        
        if train:
            data_keys = ['trainval2007', 'trainval2012']
            id_txt = 'trainval.txt'
            display_name = 'Pascal VOC 2012+2007'
        else:
            data_keys = ['test2007']
            id_txt = 'test.txt'
            display_name = 'Pascal VOC 2007 Test'
        label_path = os.path.join(root, 'voc.names') # Path to VOC class names

        # Initialize base detection dataset attributes 
            #  (scale_anchors, fmap_sizes, classes, single_resize, etc.)
        super().__init__(
            root = root, 
            label_path = label_path, 
            display_name = display_name,
            scale_anchors = scale_anchors,
            strides = strides, 
            default_input_size = default_input_size,
            ignore_threshold = ignore_threshold,
            single_augs = single_augs, 
            mosaic_augs = mosaic_augs,
            mosaic_prob = mosaic_prob, 
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
        
        if max_imgs is not None:
            assert max_imgs > 0, 'Must have `max_imgs` > 0'

            samp_idxs = random.sample(range(self.__len__()), max_imgs)
            self.img_paths = [self.img_paths[i] for i in samp_idxs]
            self.anno_paths = [self.anno_paths[i] for i in samp_idxs]

    def __len__(self) -> int:
        '''
        Gives the number of images in the dataset.
        '''
        return len(self.img_paths)
    
    def get_img(self, idx: int) -> Image.Image:
        '''
        Loads an image from the dataset (pre-transform).
        '''
        return Image.open(self.img_paths[idx]).convert('RGB')
    
    def get_anno_info(self, idx: int) -> dict:
        '''
        Parses the annotation XML file for the given index to retrieve 
        object labels and bounding box information (pre-transform).

        Returns:
            info_dict (dict): A dictionary containing:
                                - labels (torch.Tensor): Class indices for each object in the image.
                                - boxes (BoundingBoxes): Bounding boxes in (x_min, y_min, x_max, y_max) format,
                                                         scaled to the original image size.
        '''
        xml_root = ET.parse(self.anno_paths[idx]).getroot()
        size = xml_root.find('size')
        canvas_size = (int(size.find('height').text),  int(size.find('width').text))

        info_dict = {}
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

        info_dict['labels'] = torch.tensor(labels)
        info_dict['boxes'] = BoundingBoxes(torch.tensor(bboxes, dtype = torch.float32), 
                                           format = 'XYXY', canvas_size = canvas_size)
        return info_dict