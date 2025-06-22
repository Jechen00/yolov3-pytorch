#####################################
# Imports & Dependencies
#####################################
import torch
from torch.utils.data import DataLoader

from typing import Tuple, List, Union, Optional

from src.data_setup import coco_dataset, voc_dataset, transforms
from src.utils import constants


DATASETS = {
    'coco': coco_dataset.COCODataset,
    'voc': voc_dataset.VOCDataset
}


#####################################
# Functions
#####################################
def yolov3_collate_fn(batch: List[tuple]):
    '''
    Collate function for YOLOv3-style dataset.

    Args:
        batch (List[tuple]): A list of length `batch_size` containing tuples from the `__getitem__` of a `Dataset`.
                            Each tuple should have 2 elements:
                                - `imgs` (torch.Tensor): The input image of shape (num_channels, height, width)
                                - `scale_targs` (List[torch.Tensor]): List of target tensors, one per scale of the model.
                                                                      Each has shape (num_anchors, fmap_h, fmap_w, 5 + C)
    Returns:
        batch_imgs (torch.Tensor): Batched images of shape (batch_size, num_channels, height, width)
        batch_targs (List[torch.Tensor]): List of target tensors, one per scale of the model.
                                          Each has shape (batch_size, num_anchors, fmap_h, fmap_w, 5 + C)
    '''
    batch_imgs, batch_targs = zip(*batch)

    batch_imgs = torch.stack(batch_imgs)   # Shape: (batch_size, channels, height, width)

    # Group batch by scale. Each element is a tuple with batch_size tensors (on same scale)
    targs_per_scale = list(zip(*batch_targs))

    # Each element has shape (batch_size, num_anchors, fmap_h, fmap_w, 5 + C)
    batch_targs = [torch.stack(targs) for targs in targs_per_scale]
    
    return batch_imgs, batch_targs

def get_dataloaders(root: str, 
                    dataset_name: str,
                    batch_size: int, 
                    scale_anchors: List[torch.Tensor],
                    input_size: Union[int, Tuple[int, int]],
                    strides: List[Union[int, Tuple[int, int]]],
                    ignore_threshold: float = 0.5,
                    num_workers: int = 0,
                    max_imgs: Optional[Tuple[int, int]] = (None, None)) -> Tuple[DataLoader, DataLoader]:
    '''
    Creates training and validation/testing dataloaders 
    for MS-COCO (`dataset_name = 'coco'`) or Pascal VOC (`dataset_name = 'voc'`).
    If `dataset_name = 'coco'`:
        The training set uses full train2014 MS-COCO dataset and ~35k images from the val2014 MS-COCO dataset.
        The validation/test set uses ~5k images from the val2014 MS-COCO dataset.
    If `dataset_name = 'voc'`:
        The training set combines the trainval data from  Pascal VOC 2007 and 2012. 
        The validation/test set is the test data from Pascal VOC 2007.

    Args:
        root (str): Path to download datasets.
        dataset_name ('coco' or 'voc'): The dataset to use for the dataloaders.
        batch_size (int): Size used to split the datasets into batches.
        input_size (int or Tuple[int, int]): Input image size (height, width) to resize the images to.
                                             If `int`, resizing is assumed to be square.

        scale_anchors (List[torch.tensor]): List of anchor tensors for each output scale of the model.
                                            Each element has shape: (num_anchors, 2), where the last dimension gives 
                                            the (height, width) of the anchor in unit of the input size (pixels).
        strides (List[Union[int, Tuple[int, int]]]): List of strides corresponding to each scale in `scale_anchors`. 
                                                     Each stride represents the downsampling factor between the input image 
                                                     and the feature map at that scale.
        ignore_threshold (float): The IoU threshold used to encode which anchor boxes will be ignored during loss calculation.
                                  Anchor boxes that do not have the highest IoU with a ground truth box, 
                                  but have an IoU >= ignore_threshold, 
                                  will be marked with an encoded index of `-1` to indicate they should be ignored.
                                  Default value is 0.5. 

        num_workers (int): Number of workers to use for multiprocessing. Default is 0.
        max_imgs (optional, Tuple[int, int]): The maximum number of images to include for 
                                              the training dataset (`max_imgs[0]`) and validation dataset (`max_imgs[1]`).
                                              If `max_imgs[i] = None`, no maximum count is applied.

    Returns:
        train_loader (DataLoader): Dataloader for the training set.
        test_loader (DataLoader): Dataloader for the validation/test set.
    '''
    assert dataset_name in DATASETS.keys(), (
        f'`dataset` must be in {list(DATASETS.keys())}'
    )
    dataset_class = DATASETS[dataset_name]

    # Resizing and pixel rescaling will be handled inside the dataset.
    train_transforms = transforms.get_transforms(resize = False, to_float = False, train = True)
    test_transforms = transforms.get_transforms(resize = False, to_float = False, train = False)

    common_kwargs = {
        'root': root,
        'scale_anchors': scale_anchors,
        'input_size': input_size,
        'strides': strides,
        'resize': True,
        'ignore_threshold': ignore_threshold
    }
    train_dataset = dataset_class(train = True, 
                                  transforms = train_transforms,
                                  max_imgs = max_imgs[0],
                                  **common_kwargs)

    test_dataset = dataset_class(train = False, 
                                 transforms = test_transforms,
                                 max_imgs = max_imgs[1],
                                 **common_kwargs)

    # Create dataloaders
    if num_workers > 0:
        mp_context = constants.MP_CONTEXT
        persistent_workers = True
    else:
        mp_context = None
        persistent_workers = False

    train_loader = DataLoader(
        dataset = train_dataset,
        collate_fn = yolov3_collate_fn,
        batch_size = batch_size,
        shuffle = True,
        num_workers = num_workers,
        multiprocessing_context = mp_context,
        pin_memory = constants.PIN_MEM,
        persistent_workers = persistent_workers
    )

    test_loader = DataLoader(
        dataset = test_dataset,
        collate_fn = yolov3_collate_fn,
        batch_size = batch_size,
        shuffle = False,
        num_workers = num_workers,
        multiprocessing_context = mp_context,
        pin_memory = constants.PIN_MEM,
        persistent_workers = persistent_workers
    )

    return train_loader, test_loader