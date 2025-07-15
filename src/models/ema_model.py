#####################################
# Imports & Dependencies
#####################################
import torch
from torch import nn

from typing import Tuple, Dict, Any, Union
import copy


#####################################
# EMA Class
#####################################
class EMA():
    '''
    Maintains an exponential moving average (EMA) of the parameters from a model.
    Used for evaluation and inference.

    Args:
        base_model (nn.Module): The base model to track an EMA of parameters for.
        input_shape (Tuple[int]): Shape of input tensors for `base_model`. This is to initalize any lazy layers.
                                  For example, a typical CNN would be (batch_size, num_channels, height, width)
        decay (float): Decay factor for the EMA formula: `e_t = decay * x_t + (1 - decay) * e_{t-1}`,
                       where `x_t` is the current value and `e_t` is the updated EMA.
    '''
    def __init__(self, base_model: nn.Module, input_shape: Tuple[int], decay: float = 0.999):
        self.base_model = base_model
        self.decay = decay

        # Initalize any lazy layers
        device = next(base_model.parameters()).device
        dummy_X = torch.zeros(input_shape).to(device)
        with torch.no_grad():
            _ = base_model(dummy_X)

        # Copy base_model for EMA base and disable gradients
        self.ema_model = copy.deepcopy(base_model)
        self.ema_model.eval()
        for param in self.ema_model.parameters():
            param.requires_grad_(False)

    @torch.no_grad
    def update(self):
        '''
        Updates the parameters of the EMA model according to the formula:
            `e_t = decay * x_t + (1 - decay) * e_{t-1}`

        Buffers (e.g. running means and running variances) from the base model are also copied over.
        '''
        for ema_param, base_param in zip(self.ema_model.parameters(), self.base_model.parameters()):
            ema_param.mul_(self.decay).add_(base_param, alpha = 1 - self.decay)

        # Copy buffers over
        for ema_buffer, base_buffer in zip(self.ema_model.buffers(), self.base_model.buffers()):
            ema_buffer.copy_(base_buffer)

    def state_dict(self) -> Dict[str, Any]:
        '''
        Returns a state dictionary for the EMA model.
        This includes the keys:
            - model (dict): state dict of `self.ema_model`.
            - decay (float): EMA formula decay factor stored in `self.decay`.
        '''
        ema_state_dict = {
            'model': self.ema_model.state_dict(),
            'decay': self.decay
        }
        return ema_state_dict

    def load_state_dict(self, ema_state_dict, load_decay = True):
        '''
        Load EMA model weights and optionally the decay factor.

        Args:
            ema_state_dict (dict): Dictionary containing 'model' (PyTorch state_dict) and optionally 'decay' (float).
            load_decay (bool): Whether to load the decay value from `ema_state_dict`. 
                               Requires 'decay' to be in `ema_state_dict`. Default is True.
        '''
        self.ema_model.load_state_dict(ema_state_dict['model'])
        if load_decay:
            self.decay = ema_state_dict['decay']

    def compile(self, **kwargs):
        '''
        Compiles `self.ema_model` with torch.compile.
        '''
        self.ema_model.compile(**kwargs)

    def to(self, device: Union[torch.device, str]):
        '''
        Sends `self.ema_model` to the specified `device`.
        '''
        self.ema_model = self.ema_model.to(device)