#####################################
# Imports & Dependencies
#####################################
import torch
from torch import nn

from typing import Optional, List


#####################################
# Classes
#####################################
class ConvBNAct(nn.Module):
    '''
    Creates a block: convolutional layer -> optional batch normalization -> optional activation.

    Args:
        out_channels (int): Number of output channels for the conv layer.
        kernel_size (int): Kernel size for the conv layer.
        stride (int): Stride for the conv layer. Default is 1.
        padding (int): Padding for the conv layer. Default is 0.
        include_bn (bool): Whether to include batch norm after each conv layer.
                           Note that the original paper does not use batch norms. Default is False.
        activation (optional, nn.Module): Activation function applied after each conv (and batch norm if included).
                                          Default is None.
    '''
    def __init__(self, 
                 out_channels: int, 
                 kernel_size: int, 
                 stride: int = 1, 
                 padding: int = 0, 
                 include_bn: bool = False, 
                 activation: Optional[nn.Module] = None):
        super().__init__()
        include_bias = not include_bn
        layers = [nn.LazyConv2d(out_channels, kernel_size, stride, padding, bias = include_bias)]
        
        if include_bn:
            layers.append(nn.BatchNorm2d(out_channels))
            
        if activation:
            layers.append(activation)
            
        self.conv_bn_act = nn.Sequential(*layers)
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.conv_bn_act(X)
    
class Route(nn.Module):
    def __init__(self, route_layers):
        super().__init__()
        self.route_layers = route_layers
        
    def forward(self, layer_outputs: List[torch.Tensor]) -> torch.Tensor:
        '''
        Args:
            layer_outputs (List[torch.Tensor]): List of outputs from layers 0 to i-1. 
                                                Assuming that the ResConnect is layer i.
        '''
        route_outputs = [layer_outputs[i] for i in self.route_layers]
        return torch.concat(route_outputs, dim = 1)
    
class ResConnect(nn.Module):
    '''
    A residual connection (shortcut) layer.
    
    Args:
        from_layer (int): The index indicating which previous layer to add to the current input. 
                          This should be relative to this ResConnect layer, which is assumed to be layer i.
                          The current input refers to the output of layer i-1.
                          
                          Example:
                              If `from_layer = -3`, 
                              then the output of layer i-3 will be added to layer i-1.
    '''
    def __init__(self, from_layer: int):
        super().__init__()
        self.from_layer = from_layer
        
    def forward(self, layer_outputs: List[torch.Tensor]) -> torch.Tensor:
        '''
        Args:
            layer_outputs (List[torch.Tensor]): List of outputs from layers 0 to i-1. 
                                                Assuming that the ResConnect is layer i.
        '''
        return layer_outputs[-1] + layer_outputs[self.from_layer]