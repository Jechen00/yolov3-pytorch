#####################################
# Imports & Dependencies
#####################################
import torch
from torch import nn

import numpy as np
from typing import List, Tuple, Optional

from src.models import blocks
from src.utils import yolo_loader, constants


#####################################
# Functions
#####################################
def make_module(cfg_dict: str) -> Tuple[nn.Module, Optional[dict]]:
    module_name = cfg_dict['name']
    scale_info = None
    
    # Convolutional Layer
    if module_name == 'convolutional':
        if 'batch_normalize' in cfg_dict:
            include_bn = int(cfg_dict['batch_normalize']) == 1
        else:
            include_bn = False

        out_channels = int(cfg_dict['filters'])
        kernel_size = int(cfg_dict['size'])
        stride = int(cfg_dict['stride'])
        padding = kernel_size // 2 if int(cfg_dict['pad']) == 1 else 0 # 'Same' padding if pad == 1

        act_type = cfg_dict['activation']
        if act_type == 'leaky':
            activation = nn.LeakyReLU(negative_slope = 0.1, inplace = True)
        elif act_type == 'linear':
            activation = None
        else:
            raise ValueError(f'Unsupported activation type: {act_type}')

        module = blocks.ConvBNAct(
            out_channels = out_channels,
            kernel_size = kernel_size,
            stride = stride,
            padding = padding, 
            include_bn = include_bn,
            activation = activation
        )
    
    # Upsample Layer (nearest neighbor)
    elif module_name == 'upsample':
        module = nn.Upsample(scale_factor = int(cfg_dict['stride']), mode = 'nearest')
    
    # Route Layer
    elif module_name == 'route':
        route_layers = [int(i) for i in cfg_dict['layers'].split(',')]
        module = blocks.Route(route_layers = route_layers)
    
    # Shortcut/Residual Connection
    elif module_name == 'shortcut':
        # Note: all [shortcut] layers have activation: linear --> no activation
        module = blocks.ResConnect(from_layer = int(cfg_dict['from']))
    
    # YOLO Layer for scale outputs
    elif module_name == 'yolo':        
        mask = [int(i) for i in cfg_dict['mask'].split(',')]
        anchors = [float(i) for i in cfg_dict['anchors'].split(',')]
        anchors = np.array(anchors).reshape(-1, 2)

        # Get information for this scale
        scale_info = {
            'anchors': [tuple(anchor) for anchor in anchors[mask].tolist()],
            'num_classes': int(cfg_dict['classes']),
            'ignore_thresh': float(cfg_dict['ignore_thresh']),
            'truth_thresh': float(cfg_dict['truth_thresh']),
            'jitter': float(cfg_dict['jitter']),
            'random': int(cfg_dict['random'])
        }

        module = nn.Identity() # Used as a placeholder to maintain layer indices

    else:
        raise ValueError(f'Unsupported layer type: {module_name}')
        
    return module, scale_info


#####################################
# Classes
#####################################
# ------------------------
# Model Components
# ------------------------
class DarkNet53Backbone(nn.Module, yolo_loader.WeightLoadable):
    def __init__(self, cfg_file: str):
        super().__init__()
        self.model_cfgs = yolo_loader.parse_cfgs(cfg_file)
        self.module_list = nn.ModuleList()
        
        for cfg_dict in self.model_cfgs:
            module, _ = make_module(cfg_dict)
            self.module_list.append(module)

        # Check that all WeightLoadable attributes are implemented.
        self.validate_weightloadable()
                
    def forward(
        self, 
        X: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        
        Y = X
        layer_outputs = []
        for cfg_dict, module in zip(self.model_cfgs, self.module_list):
            module_name = cfg_dict['name']
            
            if module_name == 'convolutional':
                Y = module(Y)
                
            elif module_name == 'shortcut':
                Y = module(layer_outputs)
                
            layer_outputs.append(Y)
            
        return Y, layer_outputs
    
class YOLOv3Detector(nn.Module, yolo_loader.WeightLoadable):
    def __init__(self, cfg_file: str):
        super().__init__()
        self.model_cfgs = yolo_loader.parse_cfgs(cfg_file)
        self.module_list = nn.ModuleList()
        self.scale_cfgs = []
        
        for cfg_dict in self.model_cfgs:
            module, scale_cfg = make_module(cfg_dict)
            
            self.module_list.append(module)
            if scale_cfg is not None:
                self.scale_cfgs.append(scale_cfg)

        # Check that all WeightLoadable attributes are implemented.
        self.validate_weightloadable()
        
    def forward(
        self, 
        layer_outputs: List[torch.Tensor]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        '''
        layer_outputs (list[torch.Tensor]): List of layer outputs from the backbone of a YOLOv3 model.
        '''
        
        Y = layer_outputs[-1]
        scale_idx = 0
        scale_outputs = []
        
        for cfg_dict, module in zip(self.model_cfgs, self.module_list):
            module_name = cfg_dict['name']
            
            if module_name in ['convolutional', 'upsample']:
                Y = module(Y)
                
            elif module_name in ['route', 'shortcut']:
                Y = module(layer_outputs)
            
            elif module_name == 'yolo':
                scale_cfg = self.scale_cfgs[scale_idx]
                
                # Expects channels = num_anchors * (5 + C)
                batch_size, channels, height, width = Y.shape
                
                # Reshape and permute to (batch_size, num_anchors, S, S, C + 5)
                scale_Y = Y.reshape(batch_size, len(scale_cfg['anchors']), 
                                    scale_cfg['num_classes'] + 5, height, width)
                scale_Y = scale_Y.permute(0, 1, 3, 4, 2)

                scale_outputs.append(scale_Y)
                scale_idx += 1
                
            layer_outputs.append(Y)
            
        return scale_outputs
    

# ------------------------
# YOLOv3
# ------------------------
class YOLOv3(nn.Module):
    def __init__(self, 
                 backbone: nn.Module, 
                 detector_cfgs: str):
        '''
        backbone (nn.Module): The backbone/feature extractor, with weights ideally pretrained on ImageNet.
                              The output should be 2D spatial feature maps of shape (batch_size, channels, height, width).
        detector_cfgs (str): The `.cfg` file for the YOLOv3 detector configs. 
                              It should follow the structure from: 
                              https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg.
        '''
        super().__init__()
        self.backbone = backbone        
        self.detector_cfgs = detector_cfgs
        self.detector = YOLOv3Detector(cfg_file = detector_cfgs)
        self.scale_cfgs = self.detector.scale_cfgs

    def infer_scale_info(
            self, 
            input_shape: Tuple[int, int, int, int]
    ) -> Tuple[List[torch.Tensor], List[Tuple[int, int]], List[Tuple[int, int]]]:
        '''
        Infers the anchors, feature map size and stride of the model's scales, based on an input shape.

        Note: From my understanding, the original YOLOv3 paper does **not** rescale anchors to the input size,
              despite them being derived from using k-means on COCO with a 416x416 input size.
              
        Args:
            input_shape (Tuple[int, int, int, int]): Tuple for the input shape, defining dimensions 
                                                     (batch_size, channels, height, width).
                                                     Used to create a dummy input to infer feature map sizes and strides.
                                                     By YOLOv3 requirements, `height` and `width` must be divisible by 32.

        Returns:
            scale_anchors (List[Tensor]): List of anchor tensors of shape (num_anchors, 2), 
                                          where the last dimension is (anchor_w, anchor_h).
                                          Anchors are not normalized. They are in units of the input shape. 
            strides (List[Tuple[int, int]]): List of stides in format (stride_h, stride_w).
            fmap_sizes (List[Tuple[int, int]]): List of feature map sizes in format (fmap_h, fmap_w).
        '''
        _, _, height, width = input_shape
        assert (height % 32 == 0) and (width % 32 == 0), (
            'Height and width components of `input_shape` must be divisible by 32'
        ) 

        device = next(self.parameters()).device
        dummy_X = torch.zeros(input_shape).to(device)
        scale_logits = self.forward(dummy_X)

        fmap_sizes = [tuple(logits.shape[-3:-1]) for logits in scale_logits]
        strides = [(height // size[0], width // size[1]) for size in fmap_sizes]
        scale_anchors = [
            torch.tensor(scale_cfg['anchors'])
            for scale_cfg in self.scale_cfgs
        ]

        return scale_anchors, strides, fmap_sizes
    
    def init_detector_weights(self, input_shape: tuple):
        device = next(self.parameters()).device
        dummy_X = torch.zeros(input_shape).to(device)
        _ = self.forward(dummy_X) # Initalize lazy layers

        for module in self.detector.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, mean = 0.0, std = 0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
                    
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)

    def forward(self, X):
        _, layer_outputs = self.backbone(X)
        scale_outputs = self.detector(layer_outputs)
        return scale_outputs
    
class YOLOv3Full(nn.Module, yolo_loader.WeightLoadable):
    '''
    YOLOv3 model built entirely from a single config file, without separate backbone/head objects. 
    This follows the original .cfg structure from https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg.
    '''
    def __init__(self, cfg_file: str):
        super().__init__()
        full_cfgs = yolo_loader.parse_cfgs(cfg_file)
        
        self.net_cfgs = full_cfgs[0]
        self.model_cfgs = full_cfgs[1:]
        
        self.module_list = nn.ModuleList()
        self.scale_cfgs = []
        
        for cfg_dict in self.model_cfgs:
            module, scale_cfg = make_module(cfg_dict)
            
            self.module_list.append(module)
            if scale_cfg is not None:
                self.scale_cfgs.append(scale_cfg)

        # Check that all WeightLoadable attributes are implemented.
        self.validate_weightloadable()

    def infer_scale_info(
            self, 
            input_shape: Tuple[int, int, int, int]
    ) -> Tuple[List[torch.Tensor], List[Tuple[int, int]], List[Tuple[int, int]]]:
        '''
        Infers the anchors, feature map size and stride of the model's scales, based on an input shape.

        Note: From my understanding, the original YOLOv3 paper does **not** rescale anchors to the input size,
              despite them being derived from using k-means on COCO with a 416x416 input size.
              
        Args:
            input_shape (Tuple[int, int, int, int]): Tuple for the input shape, defining dimensions 
                                                     (batch_size, channels, height, width).
                                                     Used to create a dummy input to infer feature map sizes and strides.
                                                     By YOLOv3 requirements, `height` and `width` must be divisible by 32.

        Returns:
            scale_anchors (List[Tensor]): List of anchor tensors of shape (num_anchors, 2), 
                                          where the last dimension us is (anchor_w, anchor_h).
                                          Anchors are not normalized, they are in units of the input size (pixels). 
            strides (List[Tuple[int, int]]): List of stides in format (stride_h, stride_w).
            fmap_sizes (List[Tuple[int, int]]): List of feature map sizes in format (fmap_h, fmap_w).
        '''
        _, _, height, width = input_shape
        assert (height % 32 == 0) and (width % 32 == 0), (
            'Height and width components of `input_shape` must be divisible by 32'
        ) 

        device = next(self.parameters()).device
        dummy_X = torch.zeros(input_shape).to(device)
        scale_logits = self.forward(dummy_X)

        fmap_sizes = [tuple(logits.shape[-3:-1]) for logits in scale_logits]
        strides = [(height // size[0], width // size[1]) for size in fmap_sizes]
        scale_anchors = [
            torch.tensor(scale_cfg['anchors'])
            for scale_cfg in self.scale_cfgs
        ]

        return scale_anchors, strides, fmap_sizes
        
    def forward(
        self, 
        X: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        
        Y = X
        scale_idx = 0
        scale_outputs, layer_outputs = [], []
        
        for cfg_dict, module in zip(self.model_cfgs, self.module_list):
            module_name = cfg_dict['name']
            
            if module_name in ['convolutional', 'upsample']:
                Y = module(Y)
                
            elif module_name in ['route', 'shortcut']:
                Y = module(layer_outputs)
            
            elif module_name == 'yolo':
                scale_cfg = self.scale_cfgs[scale_idx]
                
                # expects channels = num_anchors * (5 + C)
                batch_size, channels, height, width = Y.shape
                
                # Reshape and permute to (batch_size, num_anchors, S, S, C + 5)
                scale_Y = Y.reshape(batch_size, len(scale_cfg['anchors']), 
                                    scale_cfg['num_classes'] + 5, height, width)
                scale_Y = scale_Y.permute(0, 1, 3, 4, 2)
                
                scale_outputs.append(scale_Y)
                scale_idx += 1
                
            layer_outputs.append(Y)
            
        return scale_outputs