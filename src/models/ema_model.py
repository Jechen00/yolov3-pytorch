#####################################
# Imports & Dependencies
#####################################
import torch
from torch import nn

from typing import Tuple
import copy


#####################################
# EMA Class
#####################################
class EMA():
    '''
    Maintains an exponential moving average of the weights from a model.
    Used for evaluation and inference.
    '''
    def __init__(self, base_model: nn.Module, input_shape: Tuple[int, int, int, int], decay: float = 0.999):
        self.base_model = base_model
        self.decay = decay

        # Initalize any lazy layers
        device = next(base_model.parameters()).device
        dummy_X = torch.zeros(input_shape).to(device)

        orig_training = base_model.training  # True if training mode, False if eval mode
        base_model.eval()
        with torch.inference_mode():
            _ = base_model(dummy_X)
        if orig_training:
            base_model.train()

        # Copy base_model for EMA base and disable gradients
        self.ema_model = copy.deepcopy(base_model)
        self.ema_model.eval()
        for param in self.ema_model.parameters():
            param.requires_grad_(False)

    @torch.no_grad
    def update(self):
        # EMA formula: e_t = decay * x_t + (1 - decay) * e_{t-1}
        for ema_param, base_param in zip(self.ema_model.parameters(), self.base_model.parameters()):
            ema_param.mul_(self.decay).add_(base_param, alpha = 1 - self.decay)

        # Copy buffers over
        for ema_buffer, base_buffer in zip(self.ema_model.buffers(), self.base_model.buffers()):
            ema_buffer.copy_(base_buffer)

    def state_dict(self):
        return self.ema_model.state_dict()

    def load_state_dict(self, state_dict):
        self.ema_model.load_state_dict(state_dict)

    def compile(self, **kwargs):
        self.ema_model.compile(**kwargs)

    def to(self, device):
        self.ema_model = self.ema_model.to(device)