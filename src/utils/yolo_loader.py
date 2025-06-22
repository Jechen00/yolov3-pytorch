#####################################
# Imports & Dependencies
#####################################
import torch
from torch import nn

import re
import numpy as np
from typing import List


#####################################
# Functions
#####################################
def parse_cfgs(cfg_file: str) -> List[dict]:
    '''
    Parses a config file for YOLOv3. 
    The format of the config file should follow: 
        https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg
    Documentation for configs:
        https://github.com/AlexeyAB/darknet/wiki/CFG-Parameters-in-the-different-layers
    '''
    file = open(cfg_file, 'r')
    raw_cfgs = re.split(r'(?=\[.*\])', file.read().strip()) # Split config file by [section]

    section_cfgs = []
    for sec_txt in raw_cfgs:
        if len(sec_txt) > 0:
            section = {}

            # Remove empty strings and comments (#)
            clean_lines = [line for line in sec_txt.split('\n') 
                           if line.strip() and line[0] != '#']

            # Creating section dictionary
            for line in clean_lines:
                if line[0] == '[':
                    section['name'] = line[1:-1] # Section name
                else:
                    key, value = line.split('=')
                    section[key.strip()] = value.strip()

            section_cfgs.append(section)
        
    return section_cfgs


#####################################
# Classes
#####################################
class WeightLoadable():
    '''
    Mix-in class that provides functions for loading weights from a
    DarkNet-format `.weights` binary file. It is not meant to be instantiated alone.

    Requirements:
        Any class that inherits from `WeightLoadable` should also inherit from `nn.Module`
        Any subclass of `WeightLoadable` must implement:
            - self.model_cfgs (List[dict]):
                A list of layer configuration dictionaries, typically parsed from a YOLO config file.
            - self.module_list (nn.ModuleList):
                A list of PyTorch modules corresponding to the model's architecture.
                Each convolutional block is expected to have the order: 
                    conv -> batch_norm (optional) -> activation (optional)
            - self.forward(x: torch.Tensor) -> torch.Tensor:
                A forward method that initializes all layers (e.g., for lazy layers).
    '''
    def validate_weightloadable(self):
        if not isinstance(self, nn.Module):
            raise TypeError('Subclass of `WeightLoadable` must also inherit from `nn.Module`')
        
        for attr in ['forward', 'model_cfgs', 'module_list']:
            if not hasattr(self, attr):
                raise AttributeError(f'Subclass of `WeightLoadable` is missing `{attr}`')
            
    def load_weights_file(self, weights_file: str, input_shape: tuple):
        '''
        Loads weights into a model using a `.weights` binary file.
        This also assumes layers are in the same order as in the original Darknet implementation.
        The loading logic follows the order of weights stored in the binary file (DarkNet-format):
            - If batch_normalize = 1 iin convolutional block:
                bn_bias -> bn_weight -> bn_running_mean -> bn_running_var -> conv_weight
            - Otherwise:
                conv_bias -> conv_weight

        Args:
            weights_file (str): Path to the `.weights` file containing pretrained weights.
            input_shape (tuple): Shape of input tensors for the model. This is to initalize any lazy layers.
                                 For example, a typical CNN would be (batch_size, num_channels, height, width)

        '''     
        device = next(self.parameters()).device
        dummy_X = torch.zeros(input_shape).to(device)
        _ = self.forward(dummy_X) # Initalize lazy layers

        with open(weights_file, 'rb') as f:
            header = np.fromfile(f, dtype = np.int32, count = 5)
            weights = np.fromfile(f, dtype = np.float32)

        curr_idx = 0 # Used to track what index we are currently on in weights
        bn_param_names = ['bias', 'weight', 'running_mean', 'running_var']
        for cfg_dict, module in zip(self.model_cfgs, self.module_list):
            if cfg_dict['name'] == 'convolutional':
                conv_bn_act = module.conv_bn_act # Get the ConvBNAct block
                conv_param_names = ['bias', 'weight']

                if 'batch_normalize' in cfg_dict:
                    include_bn = int(cfg_dict['batch_normalize']) == 1
                else:
                    include_bn = False

                # Set parameters of the batch norm layer
                if include_bn:
                    conv_param_names = ['weight'] # No biases in conv layers with batch norm
                    bn = conv_bn_act[1] # Second layer is BatchNorm2d

                    # Use no_grad to not track gradients when setting parameters
                    with torch.no_grad():
                        for param_name in bn_param_names:
                            param = getattr(bn, param_name)
                            num_param = param.numel()
                            param_weights = torch.from_numpy(weights[curr_idx:curr_idx + num_param])
                            param_weights = param_weights.to(device).type_as(param).view_as(param) # Set datatype and reshape
                            param.copy_(param_weights)

                            curr_idx += num_param

                # Set parameters of the conv layer
                conv = conv_bn_act[0] # First layer is Conv2d
                with torch.no_grad():
                    for param_name in conv_param_names:
                        param = getattr(conv, param_name)
                        num_param = param.numel()
                        param_weights = torch.from_numpy(weights[curr_idx:curr_idx + num_param])
                        param_weights = param_weights.to(device).type_as(param).view_as(param) # Set datatype and reshape
                        param.copy_(param_weights)

                        curr_idx += num_param