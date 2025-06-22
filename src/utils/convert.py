#####################################
# Functions
#####################################
import torch


#####################################
# Functions
#####################################
def corner_to_center_format(bboxes):
    '''
    Converts bbox coordinates from corner (x_min, y_min, x_max, y_max) format
    to center (x_center, y_center, width, height) format.

    In PyTorch BoundingBoxes convention: XYXY -> CXCYWH

    Args:
        bboxes: Bbox coordinates in corner format.
                Shape is (..., 4+), where the first four elements of the last dimension
                represent (x_min, y_min, x_max, y_max).

    Returns:
        torch.Tensor: The bbox coordinates converted to center format.
                      The entries after the first four elements of the last dimension 
                      are preserved and appended unchanged.
                      Shape is (..., 4+).
    '''
    x_min = bboxes[..., 0]
    y_min = bboxes[..., 1]
    x_max = bboxes[..., 2]
    y_max = bboxes[..., 3]
    
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min
    
    center_bboxes = torch.stack([x_center, y_center, width, height], dim = -1)
    return torch.concat([center_bboxes, bboxes[..., 4:]], dim = -1)

def center_to_corner_format(bboxes):
    '''
    Converts bbox coordinates from center (x_center, y_center, width, height) format
    to corner (x_min, y_min, x_max, y_max) format.

    In PyTorch BoundingBoxes convention: CXCYWH -> XYXY

    Args:
        bboxes: Bbox coordinates in center format.
                Shape is (..., 4+), where the first four elements of the last dimension
                represent (x_center, y_center, width, height).

    Returns:
        torch.Tensor: The bbox coordinates converted to corner format.
                      The entries after the first four elements of the last dimension 
                      are preserved and appended unchanged.
                      Shape is (..., 4+).
    '''
    x_center = bboxes[..., 0]
    y_center = bboxes[..., 1]
    width = bboxes[..., 2]
    height = bboxes[..., 3]
    
    x_min = x_center - width / 2
    y_min = y_center - height / 2
    x_max = x_center + width / 2
    y_max = y_center + height / 2
    
    corner_bboxes = torch.stack([x_min, y_min, x_max, y_max], dim = -1)
    return torch.concat([corner_bboxes, bboxes[..., 4:]], dim = -1)