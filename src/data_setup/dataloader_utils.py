#####################################
# Imports & Dependencies
#####################################
from __future__ import annotations

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Sampler, RandomSampler

import random, math
from typing import List, Union, Tuple, Iterable, Optional, Literal, Iterator

from src.data_setup import coco_dataset, voc_dataset, transforms
from src.utils import misc


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

def get_dataloaders(
    root: str, 
    dataset_name: Literal['coco', 'voc'],
    batch_size: int, 
    scale_anchors: List[torch.Tensor],
    strides: List[Union[int, Tuple[int, int]]],
    default_input_size: Union[int, Tuple[int, int]],
    ignore_threshold: float = 0.5,
    multi_augs: Union[Literal['mosaic', 'mixup'], List[Literal['mosaic', 'mixup']]] = 'mosaic',
    multi_aug_prob: float = 0.0,
    mixup_alpha: float = 0.5,
    multiscale_interval: Optional[int] = None,
    multiscale_sizes: Optional[List[Union[int, Tuple[int, int]]]] = None,
    max_imgs: Optional[Union[int, Tuple[int, int]]] = None,
    min_box_scale: float = 0.01,
    num_workers: int = 0,
    device: Union[torch.device, str] = 'cpu',
    return_builders = True
) -> Union[Tuple[DataLoader, DataLoader], Tuple[DataLoaderBuilder, DataLoaderBuilder]]:
    '''
    Creates training and validation/testing dataloaders for a dataset in DATASETS.
    
    If `dataset_name = 'coco'`:
        The training set uses full train2014 MS COCO dataset and ~35k images from the val2014 MS COCO dataset.
        The validation/test set uses ~5k images from the val2014 MS COCO dataset.
    If `dataset_name = 'voc'`:
        The training set combines the trainval data from  Pascal VOC 2007 and 2012. 
        The validation/test set is the test data from Pascal VOC 2007.

    Args:
        root (str): Path to download datasets, if needed.
        dataset_name (str): Dataset key indicating which dataset to use for the dataloaders. 
                            Must be a key from DATASETS.
        batch_size (int): Size used to split the datasets into batches.
        num_workers (int): Number of workers to use for multiprocessing. Default is 0.
        device (torch.device or str): The device expected to conduct training on. Default is 'cpu'.
        scale_anchors (List[torch.tensor]): List of anchor tensors for each detection scale of the model.
                                            Each element has shape: (num_anchors, 2), where the last dimension gives 
                                            the (width, height) of the anchor in unit of the input size (pixels).
        strides (List[Union[int, Tuple[int, int]]]): List of strides corresponding to each scale in `scale_anchors`. 
                                                     Each stride represents the downsampling factor between the input image 
                                                     and the feature map at that scale.
        default_input_size (int or Tuple[int, int]): A default input image size (height, width) to resize the images to.
                                                     If `int`, resizing is assumed to be square.
        ignore_threshold (float): The IoU threshold used to encode which anchor boxes will be ignored during loss calculation.
                                  Anchor boxes that do not have the highest IoU with a ground truth box, 
                                  but have an IoU >= ignore_threshold, 
                                  will be marked with `P(object) = -1` to indicate they should be ignored.
                                  Default value is 0.5. 
        multi_aug_prob (float): The probability that an image from the dataset's `__getitem__` function 
                                uses a multi-image augmentation (e.g. mosaic or mixup).
                                This is only used for the training dataset. Default is 0.0 (no multi-image augmentations)
        multiscale_interval (optional, int): Batch interval to change input image size for multiscale training.
                                             If provided, the following are also required: `multiscale_sizes`.
                                             If None, multiscale training is disabled. Default is None.
        multiscale_sizes (optional, List[Union[int, Tuple[int, int]]]): List of input sizes to use during multiscale training.
                                                                        Elements can be ints (assumed square) or (H, W) tuples.
                                                                        Example: [320, (416, 416), 608]. Default is None.
        max_imgs (optional, Union[int, Tuple[int, int]]): The maximum number of images to include for 
                                                          the training dataset (`max_imgs[0]`) and validation dataset (`max_imgs[1]`).
                                                          If `max_imgs[i] = None`, no maximum count is applied.
        min_box_scale (float): The Minimum scale of box width and height relative to the image dimensions.
                               Boxes smaller than this ratio in either dimension are discarded.
                               Default is 0.01 to represent 1% of the image width and height.
        return_builders (bool): Whether to return `DataLoaderBuilder` instances rather than the `DataLoader` instances themselves.
                                The `Dataloaders` can then be created with `DataLoaderBuilder.build()`.

    Returns:
        If `return_builders = True`:
            - train_builder (DataLoaderBuilder): The DataloaderBuilder used to construct the DataLoader for the training set.
            - test_builder (DataLoaderBuilder): The DataloaderBuilder used to construct the DataLoader for the validation/test set.
        If `return_builders = False`:
            - train_loader (DataLoader): Dataloader for the training set.
            - test_loader (DataLoader): Dataloader for the validation/test set.
    '''
    assert dataset_name in DATASETS.keys(), (
        f'`dataset` must be in {list(DATASETS.keys())}'
    )

    # Create datasets
    dataset_class = DATASETS[dataset_name]
    max_imgs = misc.make_tuple(max_imgs)

    # Resizing and pixel rescaling for single images will be handled inside the dataset.
    train_single_augs = transforms.get_single_transforms(train = True, aug_only = True)
    train_post_multi_augs = transforms.get_post_multi_transforms(aug_only = True)

    test_single_augs = transforms.get_single_transforms(train = False, aug_only = True)
    
    common_dataset_kwargs = {
        'root': root,
        'scale_anchors': scale_anchors,
        'strides': strides,
        'default_input_size': default_input_size,
        'ignore_threshold': ignore_threshold,
        'min_box_scale': min_box_scale
    }
    train_dataset = dataset_class(train = True, 
                                  single_augs = train_single_augs,
                                  post_multi_augs = train_post_multi_augs,
                                  multi_augs = multi_augs,
                                  multi_aug_prob = multi_aug_prob,
                                  mixup_alpha = mixup_alpha,
                                  max_imgs = max_imgs[0],
                                  **common_dataset_kwargs)

    test_dataset = dataset_class(train = False, 
                                 single_augs = test_single_augs,
                                 multi_aug_prob = 0.0,
                                 max_imgs = max_imgs[1],
                                 **common_dataset_kwargs)

    # Create dataloaders
    device = torch.device(device)
    if device.type == 'cuda':
        mp_context = None
        pin_memory = True
    elif device.type == 'mps':
        mp_context = 'forkserver'
        pin_memory = False
    else:
        mp_context = None
        pin_memory = False

    if num_workers > 0:
        persistent_workers = True
    else:
        mp_context = None
        persistent_workers = False

    train_sampler = MultiScaleBatchSampler(
        sampler = RandomSampler(train_dataset),
        dataset = train_dataset,
        batch_size = batch_size,
        default_input_size = default_input_size,
        drop_last = False,
        multiscale_interval = multiscale_interval,
        multiscale_sizes = multiscale_sizes
    )

    train_loader_kwargs = {
        'collate_fn': yolov3_collate_fn,
        'batch_sampler': train_sampler,
        'num_workers': num_workers,
        'multiprocessing_context': mp_context,
        'pin_memory': pin_memory,
        'persistent_workers': persistent_workers
    }

    # Multiscale isn't used for testing/validation
    test_loader_kwargs = {
        'collate_fn': yolov3_collate_fn,
        'batch_size': batch_size,
        'shuffle': False,
        'num_workers': num_workers,
        'multiprocessing_context': mp_context,
        'pin_memory': pin_memory,
        'persistent_workers': persistent_workers
    }

    train_builder = DataLoaderBuilder(dataset = train_dataset, dataloader_kwargs = train_loader_kwargs)
    test_builder = DataLoaderBuilder(dataset = test_dataset, dataloader_kwargs = test_loader_kwargs)

    if return_builders:
        return train_builder, test_builder
    else:
        train_loader = train_builder.build()
        test_loader = test_builder.build()
        return train_loader, test_loader


#####################################
# Classes
#####################################
class DataLoaderBuilder():
    '''
    A builder used to easily reproduce a DataLoader, given a dataset and kwyword arguments.
    
    Args:
        dataset (Dataset): The dataset to use for the dataloader.
        dataloader_kwargs (dict): Dictionary of keyword arguments to pass to the DataLoader when building it.
    '''
    def __init__(self, dataset: Dataset, dataloader_kwargs: dict):
        self.dataset = dataset
        self.dataloader_kwargs = dataloader_kwargs

    def build(self) -> DataLoader:
        '''
        Builds the dataloader for `dataset` using arguments from `dataloader_kwargs`.

        Returns:
            DataLoader: The dataloader for `dataset`.
        '''
        return DataLoader(dataset = self.dataset, **self.dataloader_kwargs)
    

class MultiScaleBatchSampler(Sampler):
    '''
    Batch sampler with optional multiscale training for object detection models.
    At each iteration, this sampler yields a list of `(samp_idx, input_size)` pairs,
    where `input_size` may change every `multiscale_interval` batches if enabled.

    Adapted from: https://github.com/CaoWGG/multi-scale-training/blob/master/batch_sampler.py

    Args:
        sampler (Union[Sampler[int], Iterable[int]]): Base sampler (e.g., RandomSampler or SequentialSampler)
                                                      used to sample image indices.
        dataset (Datset): The dataset of which image indices will be sampled from.
        batch_size (int): Number of sample images per batch.
        default_input_size (Union[int, Tuple[int, int]]): The input size assigned to all batches 
                                                          if multiscale training is disabled (`multiscale_interval` is not provided).
                                                          If multiscale training is enabled (`multiscale_interval` is provided),
                                                          only the inital batches of the iterator are assigned `default_input_size`,
                                                          and the rest are assigned sizes randomly sampled from `multi_scale_sizes`.
        drop_last (bool): Whether to drop the last remaining samples in a dataset if 
                          they cannot create a full batch of size `batch_size`.
                          If False, the final batch may have size <= `batch_size` ,
                          increasing the total number of batches by at most one.
                          Default is False.
        multiscale_interval (optional, int): Batch interval to change input image size for multiscale training.
                                             If provided, the following are also required: `multiscale_sizes`.
                                             If None, multiscale training is disabled and all batches are assigned `default_input_size`. 
                                             Default is None.
        multiscale_sizes (optional, List[Union[int, Tuple[int, int]]]): List of input sizes to use during multiscale training.
                                                                        Elements can be ints (assumed square) or (H, W) tuples.
                                                                        Note: for YOLO-style models, 
                                                                        these input sizes should be divisible by 32.
                                                                        Example: [320, (416, 416), 608]. Default is None.
    '''
    def __init__(
        self,
        sampler: Union[Sampler[int], Iterable[int]],
        dataset: Dataset,
        batch_size: int,
        default_input_size: Union[int, Tuple[int, int]],
        drop_last: bool = False,
        multiscale_interval: Optional[int] = None,
        multiscale_sizes: Optional[List[Union[int, Tuple[int, int]]]] = None
    ):
        if multiscale_interval is not None:
            assert multiscale_sizes is not None, (
                'If `multiscale_interval` is provided for multiscale training, `multiscale_sizes` must not be None.'
            )
            
        self.sampler = sampler
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.default_input_size = misc.make_tuple(default_input_size)
        self.multiscale_interval = multiscale_interval
        self.multiscale_sizes = [misc.make_tuple(size) for size in multiscale_sizes]
        
    def __iter__(self) -> Iterator[List[Tuple[int, Tuple[int, int]]]]:
        '''
        Creates an iterator that yields batches of `(samp_idx, input_size)` pairs.
        Provides optional multiscale training by changing `input_size` every `multiscale_interval` batches.

        Yields:
            batch (List[Tuple[int, Tuple[int, int]]]): 
                A list of (samp_idx, input_size) pairs.
                    - samp_idx (int): Index of the image sample in the dataset.
                    - input_size (Tuple[int, int]): A tuple (height, width) indicating 
                                                    the input size of the batch.
        '''
        batch = []
        num_batches = 0
        input_size = self.default_input_size

        for samp_idx in self.sampler:
            batch.append((samp_idx, input_size))
            
            if len(batch) == self.batch_size:
                yield batch
                
                num_batches += 1
                
                # Change to a new input size if using multiscale training
                change_size = (
                    (self.multiscale_interval is not None) and 
                    (num_batches % self.multiscale_interval == 0)
                )
                
                if change_size:
                    input_size = random.choice(self.multiscale_sizes)
                
                # Reset batch indices
                batch = []
                
        if (not self.drop_last) and (len(batch) > 0):
            yield batch # Yields the last batch, even if it is shorter than batch_size
            
    def __len__(self) -> int:
        '''
        Gets the number of batches in the dataset.
        '''
        if not self.drop_last:
            return math.ceil(len(self.sampler) / self.batch_size)
        else:
            return len(self.sampler) // self.batch_size