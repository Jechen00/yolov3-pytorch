#####################################
# Imports & Dependencies
#####################################
import torch
from torch import nn
from torch.optim import Optimizer, lr_scheduler

import os
from typing import Union, Dict, Optional, Any, Tuple, List
import random
import matplotlib.pyplot as plt
import numpy as np

from src.models.ema_model import EMA
from src.utils.constants import BOLD_START, BOLD_END

TrainLosses = Dict[str, List[float]]
ValLosses = Dict[str, TrainLosses]
EvalHistories = Dict[str, Dict[int, dict]]

#####################################
# Functions
#####################################
def get_colormap_colors(num_clrs, cmap_name = 'tab20'):
    cmap = plt.get_cmap(cmap_name)
    return [tuple(float(clr) for clr in cmap(i)[:3]) 
            for i in range(num_clrs)]

def make_tuple(x: Union[Any, tuple]) -> tuple:
    if not isinstance(x, tuple):
        return (x, x)
    else:
        return x
        
def set_seed(seed: int = 0):
    '''
    Sets random seed and deterministic settings 
    for reproducibility across:
        - PyTorch
        - NumPy
        - Python's random module
        - CUDA
    
    Args:
        seed (int): The seed value to set.
    '''
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.use_deterministic_algorithms(True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

def save_checkpoint(base_model: nn.Module, 
                    optimizer: Optimizer, 
                    base_train_losses: Dict[str, list],
                    val_losses: Dict[str, Dict[str, list]],
                    eval_histories: Dict[str, Dict[int, list]],
                    last_epoch: int,
                    scheduler: Optional[lr_scheduler._LRScheduler] = None, 
                    ema: Optional[EMA] = None,
                    save_dir: Optional[str] = None, 
                    checkpoint_name: Optional[str] = None,
                    save_path: Optional[str] = None):
    '''
    Saves a checkpoint containing the base model, optimizer, scheduler, and EMA model state dicts,
    along with training/validation metrics and epoch index.

    Args:
        base_model (nn.Module): The main model to save.
        optimizer (Optimizer): Optimizer used during training.
        base_train_losses (Dict[str, list]): Dictionary of lists storing train loss values per epoch for `base_model`.
        val_losses (Dict[str, Dict[str, list]]): Dictionary mapping model keys 
                                                 (e.g. 'base' for `base_model` and 'ema' for `ema`)
                                                 to a dictionary storing their validation loss values per epoch.
        eval_histories (Dict[str, Dict[int, list]]): Dictionary mapping model keys 
                                                     (e.g. 'base' for `base_model` and 'ema' for `ema`)
                                                     to a dictionary tracking evaluation metrics.
        last_epoch (int): Index of the last completed epoch.
        scheduler (optional, lr_scheduler._LRScheduler): Learning rate scheduler.
        ema (optional, EMA): An instance of the EMA class used to maintain an EMA version of the `base_model`.
        save_dir (Optional[str]): Directory to save the checkpoint.
        checkpoint_name (Optional[str]): Filename for the checkpoint (should end with '.pth' or '.pt').
        save_path (Optional[str]): Full path to save the checkpoint. 
                                   If provided, `save_dir` and `checkpoint_name` are ignored.

    '''
    if save_path is None:
        assert (save_dir is not None) and (checkpoint_name is not None), (
            'If `save_path` is not provided, both `save_dir` and `checkpoint_name` must be provided.'
        )
        # Create save path
        save_path = os.path.join(save_dir, checkpoint_name)

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok = True)

    scheduler_save = scheduler.state_dict() if scheduler is not None else None
    ema_save = ema.state_dict() if ema is not None else None

    # Create checkpoint dictionary
    checkpoint = {
        'base_model': base_model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler_save,
        'ema': ema_save,
        'base_train_losses': base_train_losses,
        'val_losses': val_losses,
        'eval_histories': eval_histories,
        'last_epoch': last_epoch
    }

    torch.save(obj = checkpoint, f = save_path)

def load_checkpoint(
    checkpoint_path: str,
    base_model: nn.Module,
    optimizer: Optimizer,
    scheduler: Optional[lr_scheduler._LRScheduler] = None,
    ema: Optional[EMA] = None,
    device: Union[str, torch.device] = 'cpu'
) -> Tuple[int, TrainLosses, ValLosses, EvalHistories]:
    '''
    Loads a saved training checkpoint from `checkpoint_path`.
    '''
    checkpoint = torch.load(checkpoint_path, map_location = device)
    base_model.load_state_dict(checkpoint['base_model'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    last_epoch = checkpoint['last_epoch']
    base_train_losses = checkpoint['base_train_losses']
    val_losses = checkpoint['val_losses']
    eval_histories = checkpoint['eval_histories']

    if scheduler is not None:
        assert checkpoint.get('scheduler') is not None, 'No scheduler state dict saved in checkpoint'
        scheduler.load_state_dict(checkpoint['scheduler'])

    if ema is not None:
        if checkpoint.get('ema') is not None:
            ema.load_state_dict(checkpoint['ema'])

        else:
            # 0.0 is filler to match lengths in val_losses['base]
            val_losses['ema'] = {key: [0.0] * len(value) for value, key in val_losses['base'].items()}
            eval_histories['ema'] = {}

            print(
                f'{BOLD_START}[NOTE]{BOLD_END} '
                'EMA provided, but no EMA state dict found in checkpoint. Continuing without loading a saved state dict...'
            )
    return last_epoch, base_train_losses, val_losses, eval_histories