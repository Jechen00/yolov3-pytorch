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
from typing import List, Optional, Union, Dict, Tuple, Any, Literal

from src import postprocess, evaluate, loss
from src.data_setup.dataloader_utils import DataLoaderBuilder
from src.utils import misc
from src.models.ema_model import EMA
from src.utils.constants import BOLD_START, BOLD_END, LOSS_NAMES, EVAL_NAMES


TrainLosses = Dict[str, List[float]]
ValLosses = Dict[str, TrainLosses]
EvalHistories = Dict[str, Dict[int, dict]]

#####################################
# Functions
#####################################
def yolov3_train_step(
    base_model: nn.Module,
    dataloader: DataLoader,
    loss_fn: loss.YOLOv3Loss,
    optimizer: Optimizer,
    scheduler: Optional[lr_scheduler._LRScheduler] = None,
    ema: Optional[EMA] = None,
    ema_update_interval: int = 1,
    accum_steps: int = 1,
    device: Union[torch.device, str] = 'cpu'
) -> Dict[str, float]:
    '''
    Performs a single traning epoch for a YOLOv3 model.
    This includes optional gradient accumulation and Exponential Moving Average (EMA) updates

    Args:
        base_model (nn.Module): The YOLOv3 model to train. This should already be on `device`.
        dataloader (Dataloader): Dataloader for the training dataset.
        loss_fn (loss.YOLOv3Loss): An instance of the YOLOv3Loss class used to compute the loss. 
                                   Gradients are computed from `loss_dict['total']` returned by its forward method.
        optimizer (Optimizer): Optimizer used to update `base_model` parameters every accumulated batch.
        scheduler (optional, lr_scheduler._LRScheduler): Learning rate scheduler. 
                                                         This updates the optimizer learning rates at every optimizer step.
                                                         Make sure that the scheduler parameters (e.g. T_max for CosineAnnealing)
                                                         account for this. 
                                                         Default is None, which disables scheduling entirely.
        ema (optional, EMA): An instance of the EMA class to maintain EMA parameters of `base_model`.
                             This should already be on `device`.
                             If not provided, no EMA parameters are computed. Default is None.
        ema_update_interval (int): The interval (in optimizer steps) to update the EMA parameters. 
                                   Default is 1 (updated every optimizer step).
        accum_steps (int): Number of batches to loop over before performing an optimizer step.
                           If `accum_steps > 1`, gradients are accumulated over multiple batches,
                           simulating a larger batch size. Default is 1.
                           See: https://lightning.ai/blog/gradient-accumulation/
        device (torch.device or str): The device to perform computations on. Default is 'cpu'.

    Returns:
        Dict[str, float]: Dictionary mapping the components of the YOLOv3 training loss
                          to its value averaged over all samples in the dataset. 
                          The keys of this dictionary are the same as the output of `loss_fn`.
    '''
    num_samps = len(dataloader.dataset)
    loss_sums = {key: 0.0 for key in loss_fn.loss_keys}
    
    full_accums = len(dataloader) // accum_steps # Number of full accumulation windows
    optimizer_steps = 0 # Counter for number of optimizer steps

    base_model.train()
    optimizer.zero_grad()
    for i, (imgs, scale_targs) in enumerate(dataloader):
        imgs = imgs.to(device)
        scale_targs = [targs.to(device) for targs in scale_targs]
        batch_size = imgs.shape[0]

        scale_logits = base_model(imgs)

        # Compute loss for batch
        loss_dict = loss_fn(scale_logits, scale_targs)

        if optimizer_steps != full_accums:
            accum_size = accum_steps
        else:
            # All full accumulation windows have been processed 
                # Last window (if any) is partial and accumulates the remainder of batch sizes
            accum_size = len(dataloader) % accum_steps

        (loss_dict['total'] / accum_size).backward() # Backpropagate only through the total loss

        for key in loss_sums:
            loss_sums[key] += loss_dict[key].detach() * batch_size

        # Simulate a larger batch_size if needed
        last_batch = ((i + 1) == len(dataloader))
        if ((i + 1) % accum_steps == 0) or last_batch:
            optimizer.step()
            optimizer.zero_grad()
            optimizer_steps += 1

            if scheduler is not None:
                # Update optimizer learning rates per optimizer step
                scheduler.step()

            # Update EMA model if needed
            if ema is not None:
                if (optimizer_steps % ema_update_interval == 0) or last_batch:
                    ema.update()

    return {key: loss_sums[key].item() / num_samps for key in loss_sums}

def yolov3_val_step(
    base_model: nn.Module, 
    dataloader: DataLoader, 
    loss_fn: loss.YOLOv3Loss, 
    ema: Optional[EMA] = None,
    should_eval: bool = False,
    scale_anchors: Optional[List[torch.Tensor]] = None,
    strides: Optional[List[Tuple[int, int]]] = None,
    obj_threshold: float = 0.01,
    nms_threshold: float = 0.5,
    map_thresholds: Optional[List[float]] = None,
    device: Union[torch.device, str] = 'cpu',
    **kwargs
) -> Dict[str, Dict[str, Any]]:
    '''
    Performs a single validation step for a YOLOv3 model. 
    This includes YOLOv3 loss computation and optional evaluation metrics (mAP and mAR)
    for both the base model and an optional exponential moving average (EMA) model.

    Args:
        base_model (nn.Module): The YOLOv3 model to compute validation loss and evalutation metrics for. 
                                This should already be on `device`.
        dataloader (DataLoader): Dataloader for the validation dataset.
        loss_fn (loss.YOLOv3Loss): An instance of the YOLOv3Loss class used to compute the loss.
        ema (optional, EMA): An EMA object tracking an EMA of `base_model` parameters.
                             If provided, validation loss and evaluation metrics are also computed for ema.ema_model.
                             This should already be on `device`.
        should_eval (bool): Whether to compute evaluation metrics (mAP and mAR). Default is False.
                            If True, must provide `strides` and `scale_anchors`.
        scale_anchors (optional, List[torch.tensor]): List of anchor tensors for each scale of the model.
                                                      Each element has shape: (num_anchors, 2), where the last dimension gives 
                                                      the (width, height) of the anchor in units of the input size (pixels). 
                                                      Default is None.
        strides (optional, List[Tuple[int, int]]): List of strides (height, width) corresponding to each scale of the model.
                                                   Default is None.
        obj_threshold (float): The porobability threshold to filter out low predicted object confidences, P(object). 
                                         Used during evaluation when computing mAP/mAR. Default is 0.01.
        nms_threshold (float): The IoU threshold used during evaluation when performing NMS for mAP/mAR. Default is 0.5.
        map_thresholds (optional, List[float]): A list of IoU thresholds used for mAP/mAR calculations.
                                                If `should_eval = True` is provided and `map_thresholds = None`, 
                                                this defaults to [0.5].
        device (torch.device or str): The device to perform computations on. Default is 'cpu'.
        **kwargs: Any other arguments for the `torchmetrics.detection.mean_ap.MeanAveragePrecision` class.
                  Note that the following arguments are overwritten: 
                    `box_format = 'xyxy'`, 'iou_thresholds = map_thresholds`.

    Returns:
        val_results (Dict[str, Dict[str, Any]]): Dictionary of validation results for each relevant model.
                                                 Keys are:
                                                    - 'base': Results for `base_model`
                                                    - 'ema': Results for `ema.ema_model` (if `ema` is provided)

                                                 Each validation result is subdictionary with the following keys:
                                                    - loss_avgs: Dictionary mapping the components of the YOLOv3 validation loss
                                                                 to its value averaged over all samples in the dataset. 
                                                                 The keys of this dictionary are the same as the output of `loss_fn`.
                                                    - eval_res: If `should_eval = True`, this is a metric dictionary (with mAP and mAR values) 
                                                                produced by `MeanAveragePrecision.compute()`. If `should_eval = False`, this is None.
                                                                For more details on the metric dictionaries, see:
                                                                    https://lightning.ai/docs/torchmetrics/stable/detection/mean_average_precision.html

    Note: Inputs (e.g. scale_anchors, strides) should have been validated externally (e.g. by a data class).
          If `should_eval = True`. This assumes all dependencies are correctly provided.
    '''
    ema_present = ema is not None
    loss_keys = loss_fn.loss_keys
    num_samps = len(dataloader.dataset)
    softmax_probs = getattr(loss_fn, 'softmax_probs', False)

    if (should_eval) and (map_thresholds is None):
        map_thresholds = [0.5]

    # -------------------------
    # Tracker Set-up
    # -------------------------
    model_trackers = {} # Store model references and tracking info for base and (optional) EMA
    model_trackers['base'] = {
        'model': base_model,
        'loss_sums': {key: 0.0 for key in loss_keys},
        'eval_res': None
    }
    if ema_present:
        model_trackers['ema'] = {
            'model': ema.ema_model,
            'loss_sums': {key: 0.0 for key in loss_keys},
            'eval_res': None
        }

    if should_eval:
        model_trackers['base']['eval_res'] = {
            'obj_threshold': obj_threshold,
            'nms_threshold': nms_threshold,
            'map_thresholds': map_thresholds
        }
        
        kwargs['box_format'] = 'xyxy'
        kwargs['iou_thresholds'] = map_thresholds
        model_trackers['base']['map_metric'] = MeanAveragePrecision(**kwargs)

        if ema_present:
            model_trackers['ema']['eval_res'] = model_trackers['base']['eval_res'].copy()
            model_trackers['ema']['map_metric'] = MeanAveragePrecision(**kwargs)

        for tracker in model_trackers.values():
            tracker['map_metric'].warn_on_many_detections = False

    # -------------------------
    # Validation Loop
    # -------------------------
    base_model.eval() # Note: ema_model should already be in eval() mode
    for imgs, scale_targs in dataloader:
        imgs = imgs.to(device)
        scale_targs = [targs.to(device) for targs in scale_targs]
        batch_size = imgs.shape[0]

        if should_eval:
            # Target bboxes are in units of the input size (pixels)
            targs_dicts = postprocess.decode_yolov3_targets(
                scale_targs = scale_targs, 
                scale_anchors = scale_anchors,
                strides = strides
            )

        for tracker in model_trackers.values():
            with torch.inference_mode():
                scale_logits = tracker['model'](imgs)

            loss_dict = loss_fn(scale_logits, scale_targs)
            for key in loss_keys:
                tracker['loss_sums'][key] += loss_dict[key] * batch_size

            if should_eval:
                # Predicted bboxes are in units of the input size (pixels)
                preds_dicts = evaluate.predict_yolov3_from_logits(
                    scale_logits = scale_logits,
                    scale_anchors = scale_anchors,
                    strides = strides,
                    obj_threshold = obj_threshold,
                    nms_threshold = nms_threshold,
                    activate_logits = True,
                    softmax_probs = softmax_probs
                )
                tracker['map_metric'].update(preds_dicts, targs_dicts)

    # -------------------------
    # Final Aggregation
    # -------------------------
    val_results = {}
    for model_key, tracker in model_trackers.items():
        if should_eval:
            map_res = tracker['map_metric'].compute() # Compute mAP and mAR values
            for key, value in map_res.items():
                # Convert tensors to floats/lists
                tracker['eval_res'][key] = value.item() if value.ndim == 0 else value.tolist()

        val_results[model_key] = {
            'loss_avgs': {key: tracker['loss_sums'][key].item() / num_samps for key in loss_keys},
            'eval_res': tracker['eval_res']
        }

    return val_results

def train(
    base_model: nn.Module,
    train_builder: DataLoaderBuilder,
    val_builder: DataLoaderBuilder,
    loss_fn: loss.YOLOv3Loss,
    optimizer: Optimizer,
    te_cfgs: TrainEvalConfigs,
    ckpt_cfgs: CheckpointConfigs,
    scheduler: Optional[lr_scheduler._LRScheduler] = None,
    ema: Optional[EMA] = None,
    device: Union[torch.device, str] = 'cpu'
) -> Tuple[TrainLosses, ValLosses, EvalHistories]:
    '''
    Trains a YOLOv3 model, tracking loss values and evaluation metrics (e.g. mAP and mAR)
    for both the base model and, if provided, an exponential moving average (EMA) model.
    Supports training from scratch or resuming from a checkpoint.
    
    The flow of each epoch is as follows:
        - Computes training loss
        - Updates the base model (using optimizer) and optionally the EMA model 
        - Optionally steps scheduler (Note that this happens either during or after the training loop)
        - Computes validation losses per epoch
        - Optionally computes mAP/mAR at evaluation epochs
        - Optionally saves checkpoint

    Args:
        base_model (nn.Module): The main YOLOv3 model to train and evaluate. Should already be on `device`.
        train_builder (DataLoaderBuilder): Builder that constructs the Dataloader for the training dataset.
        val_builder (DataLoaderBuilder): Builder that constructs the Dataloader for the validation dataset.
        loss_fn (loss.YOLOv3Loss): The YOLOv3 loss function used to compute training/validation error. 
        optimizer (Optimizer): Optimizer used to update `base_model` parameters every accumulated batch.
        te_cfgs (TrainEvalConfigs): Configuration dataclass for training and evaluation parameters.
        ckpt_cfgs (CheckpointConfigs): Configuration dataclass for saving and resuming checkpoints.
        scheduler (optional, lr_scheduler._LRScheduler): Learning rate scheduler. 
                                                         If provided and resuming from a checkpoint (`ckpt_cfgs.resume = True`),
                                                         the checkpoint file at `ckpt_cfgs.resume_path` must also include a scheduler. 
                                                         Default is None, which disables scheduling entirely — even when resuming.
        ema (optional, EMA): An instance of the EMA class used to maintain an EMA of `base_model` parameters.
                             The model at `ema.ema_model` should already be on `device`.
        device (str or torch.device): The device to perform computations on. Default is 'cpu'.

    Returns:
        base_train_losses (Dict[str, list]): Dictionary mapping loss components in `loss_fn.loss_keys` to their 
                                             list of training values per epoch. This is only for the base model (`base_model`).
        val_losses (Dict[str, Dict[str, list]]): Dictionary mapping model keys to their respective validation loss dictionary. 
                                                 Keys are:
                                                    - 'base': Validation loss dictionary for `base_model`
                                                    - 'ema': Validation loss dictionary for `ema.ema_model` (if `ema` is provided)

                                                 The validation loss dictionary is the same as `train_losses`, but for the validation dataset.
        eval_histories (dict):  Dictionary mapping model keys to their respective evaluation history dictionary.
                                Keys are:
                                - 'base': Evaluation history dictionary for `base_model`
                                - 'ema': Evaluation history dictionary for `ema.ema_model` (if `ema` is provided)

                                The evaluation history dictionary maps epoch indices (int)
                                to metric dictionaries (with mAP and mAR values) returned by `MeanAveragePrecision.compute()`. 

                                Example:
                                    {
                                        'base': {
                                            5: {'map': 0.45, 'mar_100': 0.60, ...},
                                            10: {'map': 0.50, 'mar_100': 0.65, ...}
                                        },
                                        'ema': {
                                            10: {'map': 0.52, 'mar_100': 0.66, ...}
                                        }
                                    }

                                For more details on the metric dictionaries, see:
                                    https://lightning.ai/docs/torchmetrics/stable/detection/mean_average_precision.html
    '''
    
    # -------------------------
    # Set-up
    # -------------------------
    if not hasattr(train_builder.dataset, 'multi_aug_prob'):
        raise AttributeError(
            'The dataset in `train_builder` must have a `multi_aug_prob` attribute '
            'to support disabling multi-image augmentations.'
        )
    
    if te_cfgs.multi_aug_decay_range is not None:
        orig_multi_aug_prob = train_builder.dataset.multi_aug_prob
        assert te_cfgs.multi_aug_decay_range[1] > te_cfgs.multi_aug_decay_range[0], (
            '`multi_aug_decay_range` must be in the form (start_epoch, end_epoch) where `end_epoch > start_epoch`'
        )

    # Build dataloaders for training and validation dataset
    train_loader = train_builder.build()
    val_loader = val_builder.build()

    # Dividers used for logging
    divider_len = 109
    log_divider = '-' * divider_len
    end_divider = '=' * divider_len

    if ckpt_cfgs.save_path is not None:
        print(
            f'{BOLD_START}[NOTE]{BOLD_END} Checkpoints will be saved to {ckpt_cfgs.save_path}.'
        )
    
    # Load checkpoint if needed
    if ckpt_cfgs.resume:
        checkpoint_epoch, base_train_losses, val_losses, eval_histories = misc.load_checkpoint(
            checkpoint_path = ckpt_cfgs.resume_path,
            base_model = base_model,
            optimizer = optimizer,
            scheduler = scheduler,
            ema = ema
        )

        start_epoch = checkpoint_epoch + 1
        print(
            f'{BOLD_START}[NOTE]{BOLD_END} '
            f'Successfully loaded checkpoint at {ckpt_cfgs.resume_path}. '
            f'Resuming training from epoch {start_epoch}.'
        )

    else:
        start_epoch = 0 # Starting training from scratch
        base_train_losses = {key: [] for key in loss_fn.loss_keys}
        val_losses = {
            'base': {key: [] for key in loss_fn.loss_keys}
        }
        
        # This is only used if eval_interval is not None
        eval_histories = {
            'base': {}
        }
        if ema is not None:
            val_losses['ema'] = {key: [] for key in loss_fn.loss_keys}
            eval_histories['ema'] = {}

    # Keyword arguments for the training step
    train_kwargs = dict(
        base_model = base_model, 
        dataloader = train_loader,
        loss_fn = loss_fn, 
        optimizer = optimizer,
        scheduler = None if te_cfgs.scheduler_freq == 'epoch' else scheduler,
        ema = ema,
        accum_steps = te_cfgs.accum_steps, 
        ema_update_interval = te_cfgs.ema_update_interval,
        device = device
    )

    # Start of training and evaluation
    print() # A line break between start logs and training logs
    for epoch in range(start_epoch, te_cfgs.num_epochs):
        # Construct top divider indicating epoch number
        epoch_str = f' EPOCH {epoch:>3} '
        side_len = (divider_len - len(epoch_str)) // 2
        epoch_divider = (
            '=' * side_len + 
            f'{BOLD_START}{epoch_str}{BOLD_END}' + 
            '=' * (divider_len - len(epoch_str) - side_len)
        )
        epoch_logs = [epoch_divider, log_divider] # Log messages for the epoch

        # -------------------------
        # Training
        # -------------------------
        train_start = time.time()

        # Determine if multi-image augmentation probability should decay
        decay_multi_aug_prob = (
            (te_cfgs.multi_aug_decay_range is not None) and
            (epoch >= te_cfgs.multi_aug_decay_range[0]) and 
            (train_builder.dataset.multi_aug_prob > 0)
        )
        if decay_multi_aug_prob:
            # The range of decay is [multi_aug_decay_range[0], multi_aug_decay_range[1])
            range_diff = te_cfgs.multi_aug_decay_range[1] - te_cfgs.multi_aug_decay_range[0]
            decay_progress = ((epoch + 1) - te_cfgs.multi_aug_decay_range[0]) / range_diff
            multi_aug_prob = max(0, orig_multi_aug_prob * (1 - decay_progress))

            train_builder.dataset.multi_aug_prob = multi_aug_prob # Change multi_aug_prob in the main dataset
            print(f'{BOLD_START}[NOTE]{BOLD_END} '
                  f'Multi-image augmentation probability updated to {multi_aug_prob:.3f}')

            if train_loader.persistent_workers:
                # If persistent_workers = True, rebuild the dataloader
                # This ensures each worker uses the updated multi_aug_prob attribute of the dataset
                train_loader = train_builder.build()

        # Compute average losses over batches, while updating models
        train_avgs = yolov3_train_step(**train_kwargs)
        
        # Step scheduler if learning rates should only be updated at the end of each training loop
        if (scheduler is not None) and (te_cfgs.scheduler_freq == 'epoch'):
            scheduler.step() 

        # Store and log each average loss
        base_train_log = f'|{BOLD_START}[{"BASE":<4} | {"Train Loss":<11}]{BOLD_END} '
        for loss_key in loss_fn.loss_keys:
            base_train_losses[loss_key].append(train_avgs[loss_key])
            base_train_log += f'{LOSS_NAMES[loss_key] + ":":<8} {train_avgs[loss_key]:>10.4f} | '

        epoch_logs.append(base_train_log)

        train_end = time.time()
        train_time = f'{(train_end - train_start):.2f} sec'
        time_log = (
            f'|{BOLD_START}[{"Time | ":<18}]{BOLD_END} '
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
        val_results = yolov3_val_step(
            base_model = base_model, 
            dataloader = val_loader,
            loss_fn = loss_fn,
            ema = ema,
            should_eval = should_eval,
            scale_anchors = te_cfgs.scale_anchors, 
            strides = te_cfgs.strides,
            obj_threshold = te_cfgs.obj_threshold,
            nms_threshold = te_cfgs.nms_threshold, 
            map_thresholds = te_cfgs.map_thresholds,
            device = device,
            **te_cfgs.map_kwargs
        )

        # Store and log each average validation loss
        for model_key, model_val_losses in val_losses.items():
            model_val_avgs = val_results[model_key]['loss_avgs']

            val_log = f'|{BOLD_START}[{model_key.upper():<4} | {"Val Loss":<11}]{BOLD_END} '

            for loss_key in loss_fn.loss_keys:
                model_val_losses[loss_key].append(model_val_avgs[loss_key])
                val_log += f'{LOSS_NAMES[loss_key] + ":":<8} {model_val_avgs[loss_key]:>10.4f} | '

            epoch_logs.append(val_log)
        epoch_logs.append(log_divider) # Separates loss logs from metrics and time

        # Store and log eval metrics
        if should_eval:
            for model_key, model_eval_history in eval_histories.items():
                model_eval_res = val_results[model_key]['eval_res']
                model_eval_history[epoch] = model_eval_res

                eval_log = f'|{BOLD_START}[{model_key.upper():<4} | {"Val Metrics":>11}]{BOLD_END} '
                for eval_key in ['map', 'map_large', 'map_medium', 'map_small']:
                    value = model_eval_res.get(eval_key, 0.0)
                    eval_log += f'{EVAL_NAMES[eval_key] + ":":<8} {value:>10.4f} | '
                epoch_logs.append(eval_log)

        val_end = time.time()
        val_time = f'{(val_end - val_start):.2f} sec'
        time_log += f'{"Val:":<8} {val_time:>10} | {"|":>21} {"|":>21}'
        epoch_logs.append(time_log)

        # -------------------------
        # Saving and Logs
        # -------------------------
        if ckpt_cfgs.save_path is not None:
            misc.save_checkpoint(base_model = base_model, 
                                 optimizer = optimizer, 
                                 scheduler = scheduler,
                                 ema = ema,
                                 base_train_losses = base_train_losses,
                                 val_losses = val_losses,
                                 eval_histories = eval_histories,
                                 checkpoint_epoch = epoch,
                                 save_path = ckpt_cfgs.save_path)
        
        # Print all epoch logs
        for log in epoch_logs:
            print(log)
        print(end_divider + '\n')

    return base_train_losses, val_losses, eval_histories


#####################################
# Data Classes
#####################################  
@dataclass
class TrainEvalConfigs():
    '''
    Data class for setting YOLOv3 training and evaluation configurations.

    Args:
        scale_anchors (List[torch.tensor]): List of anchor tensors for each output scale of the model.
                                            Each element has shape: (num_anchors, 2), where the last dimension gives 
                                            the (width, height) of the anchor in units of the input size (pixels). 
        strides (List[Tuple[int, int]]): List of strides (height, width) corresponding to 
                                         each scale of the model (as well as in `scale_anchors`).
        num_epochs (int): Number of epochs to train the YOLOv3 model.
        accum_steps (int): Number of batches to loop over before updating model parameters. 
                           Applies during training only. 
                           If `accum_steps > 1`, gradients are accumulated over multiple batches,
                           simulating a larger batch size. Default is 1.
                           See: https://lightning.ai/blog/gradient-accumulation/
        scheduler_freq (Literal['epoch', 'optim_step']): Defines how frequently `scheduler.step()` is called during training.
                    - epoch: `scheduler.step()` is called at the end of each training loop (once per epoch).
                    - optim_step: `scheduler.step()` is called after each optimizer step.

                    Please make sure that the arguments of the scheduler (e.g. T_max for CosineAnnealing) account for this.
                    For example:
                        - If `scheduler_freq = 'epoch'`, you may set `T_max` to the total number of epochs.
                        - If `scheduler_freq = 'optim_step'`, you may set `T_max` to the total number of optimizer steps,
                          accounting for gradient accumulation if applicable.

                    This is only used if a learning rate scheduler is provided during training. Default is `epoch`.
        ema_update_interval (int): Interval (in optimizer steps) to update EMA model weights. 
                                   This is only used if an EMA class instance is provided during training. Default is 1.
        eval_interval (optional, int): Interval (in epochs) to compute evaluation metrics on the validation dataset
                                       after the first computation at `eval_start_epoch`.
                                       If None, evaluation metrics are only computed at the every end of training. Default is None.
        eval_start_epoch (int): The epoch in which the evaluation computation periods start. 
                                This must be greater than 0 and is only used if `eval_interval` is provided.
                                Default is 0, which means evaluations happen at the start of training.
        multi_aug_decay_range (optional, Tuple[int, int]): The epoch range (start_epoch, end_epoch) 
                                                           in which to decay multi-image augmentation probability.
                                                           Note that this is a half-open interval, where decay starts at `start_epoch`
                                                           and reaches probability = 0 by `end_epoch - 1`.
                                                           Only used if multi-image augmentations are applied during training.
                                                           If not provided, multi-image augmentation probability never decays. Default is None.
        obj_threshold (optional, float): The probability threshold to filter out low predicted object confidences, P(object). 
                                         Used during evaluation when computing mAP/mAR. Default is 0.01.
        nms_threshold (optional, float): The IoU threshold used during evaluation when performing NMS for mAP/mAR. Default is 0.5.
        map_thresholds (optional, List[float]): A list of IoU thresholds used for mAP/mAR calculations.
                                                If `eval_interval` is provided and `map_thresholds = None`, this defaults to [0.5].
        map_kwargs (Dict[str, Any]): Dictionary of additional arguments to pass to the 
                                     `torchmetrics.detection.MeanAveragePrecision` class for mAP/mAR evaluation.
                                     Note: The following arguments will be overwritten: `box_format = 'xyxy'` and `iou_thresholds = map_thresholds`.
    '''
    scale_anchors: List[torch.Tensor]
    strides: List[Tuple[int, int]]

    num_epochs: int
    accum_steps: int = 1
    
    scheduler_freq: Literal['epoch', 'optim_step'] = 'epoch',
    ema_update_interval: int = 1

    eval_interval: Optional[int] = None
    eval_start_epoch: int = 0
    multi_aug_decay_range: Optional[Tuple[int, int]] = None
    obj_threshold: float = 0.01
    nms_threshold: float = 0.5
    map_thresholds: Optional[List[float]] = None
    map_kwargs: Dict[str, Any] = field(default_factory = dict)

    def __post_init__(self):
        assert self.accum_steps > 0, 'Number of accumulation steps, `accum_steps`, must be at least 1'

        if self.eval_interval is not None:
            assert self.eval_interval > 0, (
                'The interval (in epochs) for evaluation computations, `eval_interval`, must be at least 1 if provided.'
            )
            assert self.eval_start_epoch >= 0, '`eval_start_epoch`, cannot be negative.'

        # Set a default value for map_thresholds
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