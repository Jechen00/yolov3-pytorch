#####################################
# Imports & Dependencies
#####################################
import torch
from torch import nn
from torch.optim import Optimizer, lr_scheduler

import os
from typing import Union, Dict, Optional, Any
import random
import matplotlib.pyplot as plt
import numpy as np


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

def save_checkpoint(model: nn.Module, 
                    optimizer: Optimizer, 
                    train_losses: Dict[str, list],
                    val_losses: Dict[str, list],
                    eval_history: Dict[int, list],
                    last_epoch: int,
                    scheduler: Optional[lr_scheduler._LRScheduler] = None, 
                    save_dir: Optional[str] = None, 
                    checkpoint_name: Optional[str] = None,
                    save_path: Optional[str] = None):
    '''
    Saves a checkpoint containing the model, optimizer, scheduler state dicts,
    along with training/validation metrics and epoch index.

    Args:
        model (nn.Module): Model to save.
        optimizer (Optimizer): Optimizer used during training.
        train_losses (Dict[str, list]): Dictionary of lists storing train loss values per epoch.
        val_losses (Dict[str, list]): Dictionary of lists storing validation loss values per epoch.
        eval_history (Dict[int, list]): Dictionary tracking evaluation metrics.
        last_epoch (int): Index of the last completed epoch.
        scheduler (optional, lr_scheduler._LRScheduler): Learning rate scheduler.
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

    # Create checkpoint dictionary
    if scheduler is not None:
        scheduler_save = scheduler.state_dict()
    else:
        scheduler_save = None

    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler_save,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'eval_history': eval_history,
        'last_epoch': last_epoch
    }

    torch.save(obj = checkpoint, f = save_path)