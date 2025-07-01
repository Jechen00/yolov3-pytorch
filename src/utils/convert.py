#####################################
# Functions
#####################################
import torch


#####################################
# Functions
#####################################
def xyxy_to_cxcywh(bboxes):
    '''
    Converts bbox coordinates from (x_min, y_min, x_max, y_max) format
    to (x_center, y_center, width, height) format.

    In PyTorch BoundingBoxes convention: XYXY -> CXCYWH

    Args:
        bboxes: Bbox coordinates in XYXY format.
                Shape is (..., 4+), where the first four elements of the last dimension
                represent (x_min, y_min, x_max, y_max).

    Returns:
        torch.Tensor: The bbox coordinates converted to CXCYWH format.
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
    width = (x_max - x_min).clamp(min = 0)
    height = (y_max - y_min).clamp(min = 0)
    
    bboxes_cxcywh = torch.stack([x_center, y_center, width, height], dim = -1)
    return torch.concat([bboxes_cxcywh, bboxes[..., 4:]], dim = -1)

def cxcywh_to_xyxy(bboxes):
    '''
    Converts bbox coordinates from (x_center, y_center, width, height) format
    to (x_min, y_min, x_max, y_max) format.

    In PyTorch BoundingBoxes convention: CXCYWH -> XYXY

    Args:
        bboxes: Bbox coordinates in CXCYWH format.
                Shape is (..., 4+), where the first four elements of the last dimension
                represent (x_center, y_center, width, height).

    Returns:
        torch.Tensor: The bbox coordinates converted to XYXY format.
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
    
    bboxes_xyxy = torch.stack([x_min, y_min, x_max, y_max], dim = -1)
    return torch.concat([bboxes_xyxy, bboxes[..., 4:]], dim = -1)