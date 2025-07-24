#####################################
# Imports & Dependencies
#####################################
from __future__ import annotations

from torch.optim import lr_scheduler, Optimizer
from typing import List, Union


#####################################
# Learning Rate Classes
#####################################
class WarmupMultiStepLR(lr_scheduler.MultiStepLR):
    '''
    This adds a warmup period to the MultiStepLR scheduler from: 
        https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.MultiStepLR.html
    
    Args:
        optimizer (torch.optim.Optimizer): Optimizater whose learning rates will be changed by the scheduler.
        pre_warmup_lrs (float or List[float]): A list of learning rates for each parameter group 
                                               at the start of training (epoch 0).
                                               The scheduler will linearly increase the learning rate 
                                               from these values to the base learning rates over the warmup period.
                                               If provided a `float`, it is assumed that all parameter groups have the same
                                               `pre_warmup_lrs` value.
        milestones (list): List of indices for the milestone epochs to apply learning rate decay.
                        These indices must be after the value of `warmup_epochs`.
        warmup_epochs (int): Number of epochs over which to linearly increase the learning rates from 
                            pre_warmup_lrs to the base learning rates. 
                            On epoch `warmup_epochs` learning rates will reach the base learning rates.
                            If `warmup_epochs = 0`, the behavior of the scheduler will be the same as MultiStepLR.
                            Default is 5.
        gamma (float): Multiplicative factor for the learning rate decay. Default is 0.1.
        last_epoch (int): The index of last epoch. Default is -1, which indicates the start of training.
    '''
    def __init__(self, 
                 optimizer: Optimizer, 
                 pre_warmup_lrs: Union[float, List[float]],
                 milestones: List[int], 
                 warmup_epochs: int = 5,
                 gamma: float = 0.1,
                 last_epoch: int = -1):
        
        invalid = [m for m in milestones if m <= warmup_epochs]
        assert not invalid, f'Milestones {invalid} must all be after `warmup_epochs` ({warmup_epochs})'
        
        if isinstance(pre_warmup_lrs, list):
            assert len(pre_warmup_lrs) == len(optimizer.param_groups), (
                'Length of `pre_warmup_lrs` must match number of parameter groups in `optimizer`'
            )
        else:
            pre_warmup_lrs = [pre_warmup_lrs] * len(optimizer.param_groups)
        
        self.pre_warmup_lrs = pre_warmup_lrs
        self.warmup_epochs = warmup_epochs
        
        super().__init__(optimizer = optimizer, milestones = milestones, 
                         gamma = gamma, last_epoch = last_epoch) # Initializes self.base_lrs
          
    def get_lr(self):
        '''
        Returns the learning rate of each parameter group in `optimizer`.
        For the MultiStepLR scheduler component, please see:
            https://github.com/pytorch/pytorch/blob/v2.7.0/torch/optim/lr_scheduler.py#L522
        '''

        if (self.last_epoch <= self.warmup_epochs) and (self.warmup_epochs > 0):            
            # Warmup phase (Linearly changes pre_warmup_lrs to base_lrs)
                # Each warmup step is (b_lr - w_lr) / warmup_epochs
            return [
                w_lr +  self.last_epoch * (b_lr - w_lr) / self.warmup_epochs
                for w_lr, b_lr in zip(self.pre_warmup_lrs, self.base_lrs)
            ]
        else:
            return super().get_lr()
        
class WarmupCosineAnnealingLR(lr_scheduler.CosineAnnealingLR):
    '''
    This adds a warmup period to the CosineAnnealingLR scheduler from: 
        https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html
    
    Args:
        optimizer (torch.optim.Optimizer): Optimizater whose learning rates will be changed by the scheduler.
        pre_warmup_lrs (float or List[float]): A list of learning rates for each parameter group 
                                               at the start of training (epoch 0).
                                               The scheduler will linearly increase the learning rate 
                                               from these values to the base learning rates over the warmup period.
                                               If provided a `float`, it is assumed that all parameter groups have the same
                                               `pre_warmup_lrs` value.
        T_max (int): Maximum number of scheduler steps over which to anneal the learning rate with a cosine curve. 
                     The learning rate decays from `base_lr` to `eta_min` over this period.
                     After `T_max` scheduler steps, the learning rates will remain constant at `eta_min`.
                     Must be greater than `warmup_steps`. 
        eta_min (float): Minimum learning rate from the scheduler. 
                         This is the learning rate after cosine annealing stops. Default is 0.0.
        warmup_steps (int): Number of scheduler steps over which to linearly increase the learning rates from 
                            `pre_warmup_lrs` to the base learning rates. 

                            Example:
                                If the `scheduler.step()` is called only at the end of every epoch and `warmup_steps = 5`,
                                by the start of epoch 5, the learning rates will be the base learning rates.

                             If `warmup_steps = 0`, the behavior of the scheduler will be the same as CosineAnnealingLR.
                             Default is 5.
                             
        last_epoch (int): The index of last scheduler step. 
                          Note that this is a counter for the number of scheduler steps, not for the number of epochs.
                          The name is just used to match PyTorch conventions and documentation 
                          (which are a bit misleading themselves).
                          Default is -1, which indicates the start of training.
    '''
    def __init__(self, 
                 optimizer: Optimizer, 
                 pre_warmup_lrs: Union[float, List[float]],
                 T_max: int,
                 warmup_steps: int = 5,
                 eta_min: float = 0.0,
                 last_epoch: int = -1):
        
        assert T_max > warmup_steps, (
            f'Maximum number of scheduler steps `T_max` ({T_max}) must be greater than `warmup_steps` ({warmup_steps})'
        )
        
        if isinstance(pre_warmup_lrs, list):
            assert len(pre_warmup_lrs) == len(optimizer.param_groups), (
                'Length of `pre_warmup_lrs` must match number of parameter groups in `optimizer`'
            )
        else:
            pre_warmup_lrs = [pre_warmup_lrs] * len(optimizer.param_groups)
        
        self.pre_warmup_lrs = pre_warmup_lrs
        self.warmup_steps = warmup_steps
        
        super().__init__(optimizer = optimizer, T_max = T_max, 
                         eta_min = eta_min, last_epoch = last_epoch) # Initializes self.base_lrs
          
    def get_lr(self):
        '''
        Returns the learning rate of each parameter group in `optimizer`.
        '''

        if (self.last_epoch <= self.warmup_steps) and (self.warmup_steps > 0):            
            # Warmup phase (Linearly changes pre_warmup_lrs to base_lrs)
                # Each warmup change is (b_lr - w_lr) / warmup_steps
            return [
                w_lr +  self.last_epoch * (b_lr - w_lr) / self.warmup_steps
                for w_lr, b_lr in zip(self.pre_warmup_lrs, self.base_lrs)
            ]
        
        elif self.last_epoch <= self.T_max:
            return super().get_lr()
        
        else:
            # After T_max, the learning rate remains constant at eta_min
            return [
                self.eta_min
                for _ in self.optimizer.param_groups
            ]