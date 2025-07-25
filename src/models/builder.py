#####################################
# Imports & Dependencies
#####################################
import torch
from torch import nn

import numpy as np
from typing import List, Tuple, Optional, Dict

from src.models import blocks
from src.utils import yolo_loader


#####################################
# Functions
#####################################
def make_module(cfg_dict: Dict[str, str]) -> Tuple[nn.Module, Optional[dict]]:
    '''
    Creates a PyTorch module from a configuration dictionary returned by `utils.yolo_loader.parse_cfgs`.
    Currently supports the following module names ('name' key in `cfg_dict`):
        - convolutional: Convolutional layer with optional batch normalization and activation (leaky)
        - upsample: Upsample layer using nearest neighbor interpolation
        - route: Route layer
        - shortcut: Residual connection (shortcut) layer
        - yolo: YOLO detection head information. This uses a `nn.Identity` module as a placeholder
                and scale-specific settings are extracted into `scale_info`.

    Args:
        cfg_dict (Dict[str, str]): A configuration dictionary from `utils.yolo_loader.parse_cfgs`.

    Returns:
        module (nn.Module): A PyTorch module corresponding to the specified layer.
        scale_info (Optional[dict]): For 'yolo' layers, a dictionary containing:
            - 'anchors' (List[Tuple[float, float]]): Anchor box dimensions (width, height) for this scale.
            - 'num_classes' (int): Number of class labels.
            - 'ignore_thresh' (float): IoU threshold for ignoring predictions during training (used in anchor-matching).
            - 'truth_thresh' (float): IoU threshold for treating predictions as positive during training.
            - 'jitter' (float): Data augmentation jitter parameter.
            - 'random' (int): Flag for applying random resizing during training (multi-scale training).

            For more information on these keys, refer to: 
                https://github.com/AlexeyAB/darknet/wiki/CFG-Parameters-in-the-different-layers
                               
            Returns None for non-'yolo' layers.

    '''
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
    '''
    Feature extraction backbone of a DarkNet-53 model, constructed from a configuration file
    that follows the structure at: https://github.com/pjreddie/darknet/blob/master/cfg/darknet53_448.cfg

    Args:
        cfg_file (str): The full path to a `.cfg` file for DarkNet-53 feature extractor configurations.
    '''
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
        '''
        Forward function of the backbone.

        Args:
            X (torch.Tensor): Tensor input of shape (batch_size, in_channels, height, width)

        Returns:
            Y (torch.Tensor): Final output tensor after passing through all layers.
            layer_outputs (List[torch.Tensor]): List of intermediate outputs from each layer in the backbone.
        '''
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
    
class YOLOv3NeckHeads(nn.Module, yolo_loader.WeightLoadable):
    '''
    The YOLOv3 neck (feature pyramid network layers) and head (detection layers), 
    collectively referred to as the neck+heads. This is constructed from a configuration file 
    that follows the structure at: https://github.com/pjreddie/darknet/blob/master/cfg/yolov3-voc.cfg

    Args:
        cfg_file (str): Path to the `.cfg` file that defines the structure of the YOLOv3 neck+heads.
    '''
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
    ) -> List[torch.Tensor]:
        '''
        Forward function of the neck+heads.

        This function processes intermediate outputs from a backbone (e.g., DarkNet-53 feature extractor)
        and produces detection predictions at multiple scales.
        For each YOLO layer in the head component, the output of the previous layer is reshaped to
        the expected YOLOv3 format: (batch_size, num_anchors, height, width, 5 + C),
        where the last dimension represents (tx, ty, tw, th, to, class_scores).

        Args:
            layer_outputs (list[torch.Tensor]): List of layer outputs from the backbone of a YOLOv3 model.

        Returns:
            scale_outputs (List[torch.Tensor]): List of prediction tensors, one per detection scale.
                                                The shape of each tensor is (batch_size, num_anchors, height, width, 5 + C).
        '''
        
        layer_outputs = list(layer_outputs)
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
                
                # Reshape and permute to (batch_size, num_anchors, fmap_height, fmap_width, 5 + C)
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
                 neck_heads_cfg: str):
        '''
        The complete YOLOv3 model, including the backbone, neck, and heads.
        The model components are grouped as follows:
            - backbone: a feature extractor (any nn.Module)
            - neck_heads: the feature pyramid network (FPN) and detection layers, 
                          built from a configuration file

        Args:
            backbone (nn.Module): The backbone/feature extractor, with weights ideally pretrained on ImageNet.
                                  The output should be 2D spatial feature maps of shape (batch_size, channels, height, width).
            neck_heads_cfg (str): Full path to a `.cfg` file for the YOLOv3 neck and heads configurations. 
                                 It should follow the structure from: 
                                    https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg.
        '''
        super().__init__()
        self.backbone = backbone        
        self.neck_heads_cfg = neck_heads_cfg
        self.neck_heads = YOLOv3NeckHeads(cfg_file = neck_heads_cfg)
        self.scale_cfgs = self.neck_heads.scale_cfgs

    def infer_scale_info(
            self, 
            input_shape: Tuple[int, int, int, int]
    ) -> Tuple[List[torch.Tensor], List[Tuple[int, int]], List[Tuple[int, int]]]:
        '''
        Infers the feature map size and stride of the model's scales, based on an input shape.

        Note: From my understanding, the original YOLOv3 paper does **not** rescale anchors to the input size,
              despite them being derived from using k-means on COCO with a 416x416 input size.
              
        Args:
            input_shape (Tuple[int, int, int, int]): Tuple for the input shape,
                                                     defining dimensions (batch_size, channels, height, width).
                                                     Used to create a dummy input to infer feature map sizes and strides.
                                                     By YOLOv3 requirements, `height` and `width` must be divisible by 32.

        Returns:
            scale_anchors (List[Tensor]): List of anchor tensors of shape (num_anchors, 2), 
                                          where the last dimension is (anchor_w, anchor_h).
                                          Anchors are not normalized. They are in units of the input shape (pixels). 
            strides (List[Tuple[int, int]]): List of strides in format (stride_h, stride_w).
            fmap_sizes (List[Tuple[int, int]]): List of feature map sizes in format (fmap_h, fmap_w).
        '''
        _, _, height, width = input_shape
        assert (height % 32 == 0) and (width % 32 == 0), (
            'Height and width components of `input_shape` must be divisible by 32'
        ) 

        device = next(self.parameters()).device
        dummy_X = torch.zeros(input_shape).to(device)
        
        orig_training = self.training
        self.eval() 
        with torch.no_grad():
            scale_logits = self.forward(dummy_X)
        if orig_training:
            self.train()

        fmap_sizes = [tuple(logits.shape[-3:-1]) for logits in scale_logits]
        strides = [(height // size[0], width // size[1]) for size in fmap_sizes]
        scale_anchors = [
            torch.tensor(scale_cfg['anchors'])
            for scale_cfg in self.scale_cfgs
        ]

        return scale_anchors, strides, fmap_sizes
    
    def init_neck_heads(self, input_shape: tuple):
        '''
        Randomly initalizes the neck and heads layers as follows:
            - Convolutional layers:
                - Weights are initialized from a normal distribution, N(mu = 0, sigma = 0.01)
                - Biases are set to 0.0
            - Batch normalization layers:
                - Weights are set to 1.0
                - Biases are set to 0.0
        '''
        device = next(self.parameters()).device
        dummy_X = torch.zeros(input_shape).to(device)

        orig_training = self.training
        self.eval() 
        with torch.no_grad():
            _ = self.forward(dummy_X) # Initalize lazy layers
        if orig_training:
            self.train()

        for module in self.neck_heads.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, mean = 0.0, std = 0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
                    
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)

    def forward(self, X: torch.Tensor) -> List[torch.Tensor]:
        '''
        Forward function of the YOLOv3 model.
        
        Args:
            X (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            scale_outputs (List[torch.Tensor]): List of prediction tensors, one per detection scale.
                                                The shape of each tensor is 
                                                (batch_size, num_anchors, height, width, 5 + C).
        '''
        _, layer_outputs = self.backbone(X)
        scale_outputs = self.neck_heads(layer_outputs)
        return scale_outputs
    
class YOLOv3Full(nn.Module, yolo_loader.WeightLoadable):
    '''
    YOLOv3 model built entirely from a single configuration file, without separate objects for the backbone and neck+heads.
    This follows the original `.cfg` structure from https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg.

    Args:
        cfg_file: Path to a `.cfg` file for the full YOLOv3 model configs.
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
        Infers the feature map size and stride of the model's scales, based on an input shape.

        Note: From my understanding, the original YOLOv3 paper does **not** rescale anchors to the input size,
              despite them being derived from using k-means on COCO with a 416x416 input size.
              
        Args:
            input_shape (Tuple[int, int, int, int]): Tuple for the input shape,
                                                     defining dimensions (batch_size, channels, height, width).
                                                     Used to create a dummy input to infer feature map sizes and strides.
                                                     By YOLOv3 requirements, `height` and `width` must be divisible by 32.

        Returns:
            scale_anchors (List[Tensor]): List of anchor tensors of shape (num_anchors, 2), 
                                          where the last dimension is (anchor_w, anchor_h).
                                          Anchors are not normalized. They are in units of the input shape (pixels). 
            strides (List[Tuple[int, int]]): List of strides in format (stride_h, stride_w).
            fmap_sizes (List[Tuple[int, int]]): List of feature map sizes in format (fmap_h, fmap_w).
        '''
        _, _, height, width = input_shape
        assert (height % 32 == 0) and (width % 32 == 0), (
            'Height and width components of `input_shape` must be divisible by 32'
        ) 

        device = next(self.parameters()).device
        dummy_X = torch.zeros(input_shape).to(device)
        
        orig_training = self.training
        self.eval() 
        with torch.no_grad():
            scale_logits = self.forward(dummy_X)
        if orig_training:
            self.train()

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
    ) -> List[torch.Tensor]:
        '''
        Forward function of the YOLOv3 model.
        
        Args:
            X (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            scale_outputs (List[torch.Tensor]): List of prediction tensors, one per detection scale.
                                                The shape of each tensor is 
                                                (batch_size, num_anchors, height, width, 5 + C).
        '''
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