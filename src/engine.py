#####################################
# Imports & Dependencies
#####################################
from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler, Optimizer
from torchmetrics.detection.mean_ap import MeanAveragePrecision

import os
import time
from dataclasses import dataclass, field
from typing import List, Optional, Union, Dict, Tuple, Any

from src import postprocess, evaluate
from src.data_setup.dataloader_utils import DataLoaderBuilder
from src.data_setup.dataset_utils import DetectionDatasetBase
from src.utils import misc
from src.utils.constants import BOLD_START, BOLD_END, LOSS_NAMES, EVAL_NAMES


#####################################
# Functions
#####################################
def yolov3_train_step(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: Optimizer,
    accum_steps: int = 1,
    device: Union[torch.device, str] = 'cpu'
) -> Dict[str, float]:
    num_samps = len(dataloader.dataset)
    loss_sums = {key: 0.0 for key in loss_fn.loss_keys}
    
    model.train()
    for i, (imgs, scale_targs) in enumerate(dataloader):
        print(f'BATCH {i}: IMAGE SHAPE: {imgs.shape}')
        print(f'BATCH {i}: TARGET SHAPES: {[targs.shape for targs in scale_targs]}')
        imgs = imgs.to(device)
        scale_targs = [targs.to(device) for targs in scale_targs]
        batch_size = imgs.shape[0]

        scale_logits = model(imgs)

        # Compute loss for batch
        loss_dict = loss_fn(scale_logits, scale_targs)
        (loss_dict['total'] / accum_steps).backward() # Backpropagate only through the total loss

        for key in loss_sums:
            loss_sums[key] += loss_dict[key].detach() * batch_size

        # Used to simulate a larger batch_size
        num_batches = i + 1
        if (num_batches % accum_steps == 0) or (num_batches == len(dataloader)):
            optimizer.step()
            optimizer.zero_grad()

    return {key: loss_sums[key].item() / num_samps for key in loss_sums}

def yolov3_val_step(
    model: nn.Module, 
    dataloader: DataLoader, 
    loss_fn: nn.Module, 
    should_eval: bool = False,
    scale_anchors: Optional[List[torch.Tensor]] = None,
    strides: Optional[List[Tuple[int, int]]] = None,
    obj_threshold: Optional[float] = None,
    nms_threshold: Optional[float] = None,
    map_thresholds: Optional[List[float]] = None,
    device: Union[torch.device, str] = 'cpu',
    **kwargs
) -> Tuple[dict, Optional[dict]]:
    '''
    Args:
        **kwargs: Any other arguments for the `torchmetrics.detection.mean_ap.MeanAveragePrecision` class.
                  Note that the following arguments are overwritten: 
                    `box_format='xyxy'`, 'iou_thresholds=map_thresholds`.
    '''

    # Note: Inputs (e.g. scale_anchors, strides) should have been validated externally.
        # If `should_eval=True`. This assumes all dependencies are correctly provided.
    if should_eval:
        eval_res = {
            'obj_threshold': obj_threshold,
            'nms_threshold': nms_threshold,
            'map_thresholds': map_thresholds
        }
        
        kwargs['box_format'] = 'xyxy'
        kwargs['iou_thresholds'] = map_thresholds
        map_metric = MeanAveragePrecision(**kwargs)

    num_samps = len(dataloader.dataset)
    loss_sums = {key: 0.0 for key in loss_fn.loss_keys}
    softmax_probs = getattr(loss_fn, 'softmax_probs', False)

    model.eval()
    for imgs, scale_targs in dataloader:
        imgs = imgs.to(device)
        scale_targs = [targs.to(device) for targs in scale_targs]
        batch_size = imgs.shape[0]

        with torch.inference_mode():
            scale_logits = model(imgs)

        # -------------------------
        # Loss
        # -------------------------
        # Compute loss for the batch
        loss_dict = loss_fn(scale_logits, scale_targs)

        for key in loss_sums:
            loss_sums[key] += loss_dict[key] * batch_size

        # -------------------------
        # Evaluation Metrics
        # -------------------------
        if should_eval:
            # Target and prediction bboxes are in units of the input size (pixels)
            targs_dicts = postprocess.decode_yolov3_targets(
                scale_targs = scale_targs, 
                scale_anchors = scale_anchors,
                strides = strides
            )
            preds_dicts = evaluate.predict_yolov3_from_logits(
                scale_logits = scale_logits,
                scale_anchors = scale_anchors,
                strides = strides,
                obj_threshold = obj_threshold,
                nms_threshold = nms_threshold,
                activate_logits = True,
                softmax_probs = softmax_probs
            )
            map_metric.update(preds_dicts, targs_dicts)

    loss_avgs = {key: loss_sums[key].item() / num_samps for key in loss_sums}
    if should_eval:
        map_res = map_metric.compute() # Compute mAP and mAR values
        for key, value in map_res.items():
            # Convert tensors to floats/lists
            eval_res[key] = value.item() if value.ndim == 0 else value.tolist()

        return loss_avgs, eval_res
    else:
        return loss_avgs, None
    
def train(
    model: nn.Module,
    train_builder: DataLoaderBuilder,
    val_builder: DataLoaderBuilder,
    loss_fn: nn.Module,
    optimizer: Optimizer,
    te_cfgs: TrainEvalConfigs,
    ckpt_cfgs: CheckpointConfigs,
    scheduler: Optional[lr_scheduler._LRScheduler] = None,
    device: Union[torch.device, str] = 'cpu'
) -> Tuple[dict, dict, dict]:
    '''
    Trains a YOLOv3 model, tracking loss values and evaluation metrics (e.g. mAP and mAR).
    Supports training from scratch or resuming from a checkpoint.
    
    The flow of each epoch is as follows:
        - Computes training loss and updates the model (per accumulated batch)
        - Computes validation loss per epoch
        - Optionally computes mAP/mAR at evaluation epochs
        - Optionally saves model checkpoints
    '''
    
    # -------------------------
    # Setup & Initialization
    # -------------------------
    assert hasattr(train_builder.dataset, 'mosaic_prob'), (
        'The dataset in `train_builder` must have a `mosaic_prob` attribute to support disabling mosaic augmentations.'
    )
    assert hasattr(val_builder.dataset, 'mosaic_prob'), (
        'The dataset in `val_builder` must have a `mosaic_prob` attribute to support disabling mosaic augmentations.'
    )

    train_loader = train_builder.build()
    val_loader = val_builder.build()

    log_divider = '-' * 114
    epoch_divider = '=' * 114
    start_logs = [] # Log messages to print prior to training/evaluation
    if ckpt_cfgs.save_path is not None:
        start_logs.append(
            f'{BOLD_START}[NOTE]{BOLD_END} ' 
            f'Checkpoints will be saved to {ckpt_cfgs.save_path}.'
        )
    
    # Load in previous checkpoint if resuming training
    if ckpt_cfgs.resume:
        checkpoint = torch.load(ckpt_cfgs.resume_path, map_location = device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])

        if scheduler is not None:
            assert checkpoint['scheduler'] is not None, 'No scheduler state dict saved in checkpoint'
            scheduler.load_state_dict(checkpoint['scheduler'])
        
        last_epoch = checkpoint['last_epoch']
        train_losses = checkpoint['train_losses']
        val_losses = checkpoint['val_losses']
        eval_history = checkpoint['eval_history']
            
        start_logs.append(
            f'{BOLD_START}[NOTE]{BOLD_END} '
            f'Successfully loaded checkpoint at {ckpt_cfgs.resume_path}. '
            f'Resuming training from epoch {last_epoch + 1}.'
        )

    else:
        last_epoch = -1
        train_losses = {key: [] for key in loss_fn.loss_keys}
        val_losses = {key: [] for key in loss_fn.loss_keys}
        eval_history = {} # This is only used if eval_interval is not None
        
    for log in start_logs:
        print(log)
    print()  # Add a blank line before training/evaluation logs
    
    # Start of training and evaluation
    for epoch in range(last_epoch + 1, te_cfgs.num_epochs):
        epoch_logs = [] # Log messages for the epoch

        # -------------------------
        # Training
        # -------------------------
        train_start = time.time()

        # Disable mosaic for the last 20% of epochs
        # This lets the model fine-tune to more realistic, single images
        if (epoch == int(0.8 * te_cfgs.num_epochs)) and (train_builder.dataset.mosaic_prob > 0):
            train_builder.dataset.mosaic_prob = 0 # Set mosaic_prob to 0 in the main dataset

            if train_loader.persistent_workers:
                # If persistent_workers = True, rebuild the dataloader
                # This ensures each worker uses the updated mosaic_prob prob attribute of the dataset
                train_loader = train_builder.build()
            print(f'{BOLD_START}[NOTE]{BOLD_END} Mosaic augmentation has been disabled from this point forward.')

        # Compute average losses over batches
        train_avgs = yolov3_train_step(
            model = model, 
            dataloader = train_loader,
            loss_fn = loss_fn, 
            optimizer = optimizer,
            accum_steps = te_cfgs.accum_steps, 
            device = device
        )

        # Update optimizer learning rates
        if scheduler is not None:
            scheduler.step() 

        # Store and log each average loss
        train_log = f'|{BOLD_START}[EPOCH {epoch:>3} | {"Train Loss":<11}]{BOLD_END} '
        for key in loss_fn.loss_keys:
            train_losses[key].append(train_avgs[key])
            train_log += f'{LOSS_NAMES[key] + ":":<8} {train_avgs[key]:>10.4f} | '

        epoch_logs.append(train_log)

        train_end = time.time()

        train_time = f'{(train_end - train_start):.2f}' + ' sec'
        time_log = (
            f'|{BOLD_START}[EPOCH {epoch:>3} | {"Time":<11}]{BOLD_END} '
            f'{"Train:":<8} {train_time:>10} | '
        )

        # -------------------------
        # Validation
        # -------------------------
        val_start = time.time()

        # Evaluate metrics (mAP) at specified intervals and at the final epoch
        should_eval = False
        if epoch == (te_cfgs.num_epochs - 1):
            should_eval = True
        elif (te_cfgs.eval_interval is not None) and (epoch >= te_cfgs.eval_start_epoch):
            if (epoch - te_cfgs.eval_start_epoch) % te_cfgs.eval_interval == 0:
                should_eval = True

        # Compute average losses over batches and eval metrics
            # eval_res is None if should_eval is False
        val_avgs, eval_res = yolov3_val_step(
            model = model, 
            dataloader = val_loader,
            loss_fn = loss_fn,
            should_eval = should_eval,
            scale_anchors = te_cfgs.scale_anchors, 
            strides = te_cfgs.strides,
            obj_threshold = te_cfgs.obj_threshold,
            nms_threshold = te_cfgs.nms_threshold, 
            map_thresholds = te_cfgs.map_thresholds,
            device = device,
            **te_cfgs.map_kwargs
        )

        # Store and log each average loss
        val_log = f'|{BOLD_START}[EPOCH {epoch:>3} | {"Val Loss":<11}]{BOLD_END} '
        for key in loss_fn.loss_keys:
            val_losses[key].append(val_avgs[key])
            val_log += f'{LOSS_NAMES[key] + ":":<8} {val_avgs[key]:>10.4f} | '

        epoch_logs.append(val_log)
        epoch_logs.append(log_divider) # Separates loss logs from val and time

        # Store and log eval metrics
        if should_eval:
            eval_history[epoch] = eval_res

            eval_log = f'|{BOLD_START}[EPOCH {epoch:>3} | {"Val Metrics":<11}]{BOLD_END} '
            for key in ['map', 'map_large', 'map_medium', 'map_small']:
                eval_log += f'{EVAL_NAMES[key] + ":":<8} {eval_res[key]:>10.4f} | '
            epoch_logs.append(eval_log)

        val_end = time.time()

        val_time = f'{(val_end - val_start):.2f}' + ' sec'
        time_log += f'{"Val:":<8} {val_time:>10} | {"|":>21} {"|":>21}'
        epoch_logs.append(time_log)

        # -------------------------
        # Saving and Logs
        # -------------------------
        if ckpt_cfgs.save_path is not None:
            misc.save_checkpoint(model = model, 
                                 optimizer = optimizer, 
                                 scheduler = scheduler,
                                 train_losses = train_losses,
                                 val_losses = val_losses,
                                 eval_history = eval_history,
                                 last_epoch = epoch,
                                 save_path = ckpt_cfgs.save_path)
        print(epoch_divider)
        for log in epoch_logs:
            print(log)
        print(epoch_divider + '\n')

    return train_losses, val_losses, eval_history


#####################################
# Data Classes
#####################################  
@dataclass
class TrainEvalConfigs():
    '''
    Data class for setting YOLOv3 training and evaluation configurations.

    Args:
        num_epochs (int): Number of epochs to train the YOLOv3 model.
        accum_steps (int): Number of batches to loop over before updating model parameters. 
                           Applies during training only. 
                           If `accum_steps > 1`, gradients are accumulated over multiple batches,
                           simulating a larger batch size. Default is 1.
                           See: https://lightning.ai/blog/gradient-accumulation/
        eval_interval (optional, int): Interval (in epochs) to compute evaluation metrics on the validation dataset
                                       after the first computation at `eval_start_epoch`.
                                       If None, evaluation metrics are never computed.
                                       If provided, the following are also required: 
                                       `scale_anchors`, `strides`, `obj_threshold`, `nms_threshold`. Default is None.
        eval_start_epoch (int): The epoch in which the evaluation computation periods start. 
                                This is must be greater than 0 and is only used if `eval_interval` is provided.
                                Default is 0, which means evaluations happen at the start of training.
        scale_anchors (optional, List[torch.tensor]): List of anchor tensors for each output scale of the model.
                                                      Each element has shape: (num_anchors, 2), where the last dimension gives 
                                                      the (width, height) of the anchor in units of the input size (pixels). 
                                                      Default is None.
        strides (optional, List[Tuple[int, int]]): List of strides (height, width) corresponding to each scale from `scale_anchors`.
                                                   Default is None.
        obj_threshold (optional, float): Threshold to filter out low predicted object probabilities, i.e. P(object). 
                                         Used during evaluation when computing mAP/mAR. Default is None.
        nms_threshold (optional, float): The IoU threshold used during evaluation when performing NMS for mAP/mAR. Default is None.
        map_thresholds (optional, List[float]): A list of IoU thresholds used for mAP/mAR calculations.
                                                If `eval_interval` is provided and `map_thresholds=None`, this defaults to [0.5].
        map_kwargs (Dict[str, Any]): Dictionary of additional arguments to pass to the 
                                     `torchmetrics.detection.MeanAveragePrecision` class for mAP/mAR evaluation.
                                     Note: The following arguments will be overwritten:
                                     `box_format='xyxy'` and `iou_thresholds=map_thresholds`.
    '''
    num_epochs: int
    accum_steps: int = 1
    
    eval_interval: Optional[int] = None
    eval_start_epoch: int = 0
    scale_anchors: Optional[List[torch.Tensor]] = None
    strides: Optional[List[Tuple[int, int]]] = None
    obj_threshold: Optional[float] = None
    nms_threshold: Optional[float] = None
    map_thresholds: Optional[List[float]] = None
    map_kwargs: Dict[str, Any] = field(default_factory = dict)

    def __post_init__(self):
        assert self.accum_steps > 0, 'Number of accumulation steps, `accum_steps`, must be at least 1'

        # Set a default value for map_thresholds
        if self.eval_interval is not None:
            assert self.eval_interval > 0, 'The interval (in epochs) for evaluation computations, `eval_interval`, must be at least 1 if provided.'
            assert self.eval_start_epoch >= 0, '`eval_start_epoch`, cannot be negative.'
            for eval_attr in ['scale_anchors', 'strides', 'obj_threshold', 'nms_threshold']:
                assert getattr(self, eval_attr) is not None, f'If `eval_interval` is provided for evaluations, `{eval_attr}` must not be None'
            self.map_thresholds = [0.5] if self.map_thresholds is None else self.map_thresholds
        
@dataclass
class CheckpointConfigs():
    '''
    Data class for setting checkpoint saving and resuming configurations.

    Args:
        save_dir (optional, str): Directory to save checkpoint every epoch.
                                  Required if `checkpoint_name` is provided.
                                  If `save_dir` and `checkpoint_name` are None, checkpoints will not be saved.
        checkpoint_name (optional, str): File name for the checkpoint. 
                                         If missing an extension (.pt or .pth), `.pth` will be appended.
                                         If only `save_dir` is provided, defaults to `checkpoint.pth`.
        ignore_exists (bool): Whether to ignore existing checkpoint file at `save_dir/checkpoint_name`.
                              If `False` and a file already exists, training is halted unless `resume = True`.
        resume_path (optional, str): Full path to a checkpoint file to resume training from.
                                     If not provided and `resume = True`, defaults to `save_dir/checkpoint_name`.
        resume (bool): Whether to resume training from a previous checkpoint.
    '''
    save_dir: Optional[str] = None
    checkpoint_name: Optional[str] = None
    ignore_exists: bool = False
    resume_path: Optional[str] = None
    resume: bool = False
    
    def __post_init__(self):
        # Get the save_path
        match (self.save_dir, self.checkpoint_name):
            case (None, None):
                self.save_path = None # No saving needed

            case (str(), str()):
                # Add .pth if checkpoint_name doesn't end with .pth or .pts
                if not self.checkpoint_name.endswith(('.pth', '.pt')):
                    self.checkpoint_name += '.pth'
                self.save_path = os.path.join(self.save_dir, self.checkpoint_name)

            case (str(), None):
                # Set a default file name for saved checkpoint
                self.save_path =  os.path.join(self.save_dir, 'checkpoint.pth')

            case (None, str()):
                raise ValueError('`save_dir` must be a specified string if `checkpoint_name` is given.')
                
        # Check if resuming and if a resume_path needs to be set
        if self.resume and (self.resume_path is None):
            assert self.save_path is not None, (
                'Cannot resume training. Neither `resume_path` is provided, '
                'nor both `save_dir` and `checkpoint_name`.'
            )
            # Use save_path as resume_path if none was provided explicitly
            self.resume_path = self.save_path
        
        # If not resuming, check if save_path already has an existing file
        elif (not self.resume) and (self.save_path is not None):
            if os.path.isfile(self.save_path) and (not self.ignore_exists):
                raise FileExistsError(
                    f'A file already exists at `save_path`: {self.save_path}, but `resume = False`. '
                    f'To allow overwriting this file and start training from scratch, set `ignore_exists = True`.'
                )