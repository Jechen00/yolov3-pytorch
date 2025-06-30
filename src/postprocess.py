#####################################
# Imports & Dependencies
#####################################
import torch

from typing import List, Tuple, Union, Optional, Literal
BBoxConfProbs = Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]

from src.utils import convert


#####################################
# Functions
#####################################
def activate_yolov3_logits(
    scale_logits: List[torch.Tensor], 
    split_preds: bool = False,
    softmax_probs: bool = False
) -> Union[List[torch.Tensor], BBoxConfProbs]:
    '''
    Applies activation functions to YOLOv3 logits for each scale.
    
    Args:
        scale_logits (List[Tensor]): Scale logit tensors from YOLOv3, one per scale.
                                     Each has shape (batch_size, num_anchors, fmap_h, fmap_w, 5 + C)
        split_preds (bool): Whether to split outputs into bbox, obj, class components.
                            If `split_preds = True`, this function returns a tuple of 3 lists of tensors, 
                            rather than a single list of tensors
        softmax_probs (bool): Whether to use softmax over classes instead of sigmoid.
    '''
    
    def activate(logits: torch.Tensor) -> torch.Tensor:
        logits = logits.clone()

        # Apply sigmoid to tx, ty (xy-offsets) and to (objectness)
        logits[..., [0, 1, 4]] = torch.sigmoid(logits[..., [0, 1, 4]])

        if not softmax_probs:
            logits[..., 5:] = torch.sigmoid(logits[..., 5:])
        else:
            logits[..., 5:] = torch.softmax(logits[..., 5:], dim = -1) 

        return logits

    if not split_preds:
        return [activate(logits) for logits in scale_logits]
    
    else:
        bbox_preds, conf_preds, class_preds = [], [], []
        for logits in scale_logits:
            preds = activate(logits)
            bbox_preds.append(preds[..., :4])
            conf_preds.append(preds[..., 4])
            class_preds.append(preds[..., 5:])
        
        return bbox_preds, conf_preds, class_preds
    
def decode_yolov3_bboxes(bboxes: torch.Tensor, 
                         anchors: torch.Tensor, 
                         stride: Tuple[int, int],
                         mask: Optional[torch.Tensor] = None,
                         return_format: Literal['xyxy', 'cxcywh'] = 'cxcywh',
                         return_units: Literal['pixel', 'fmap'] = 'pixel') -> torch.Tensor:
    '''
    bboxes shape: (batch_size, num_anchors, fmap_h, fmap_w, 4+)
    anchors torch.tensor: Anchors of shape (num_anchors, 2), where the last dimension gives 
                          the (width, height) of the anchors in units of the input size (pixels).
                          These anchors would correspond to the feature map scale of `bboxes`.
    stride (Tuple[int, int]): The height and width stride corresponding to the feature map scale of `bboxes`.
    mask shape: (batch_size, num_anchors, fmap_h, fmap_w)
    return_format ('xyxy' or 'cxcywh'): Return format of the bounding boxes after decoding. Default is 'cxcywh'.
    return_units ('pixel' or 'fmap'): Units for decoded bounding boxes.
                        - 'pixel' returns coordinates in units of the input size.
                        - 'fmap' returns normalized coordinates relative to feature map, i.e., divides pixel coords by stride.
                        Default is 'pixel'.

    Note: mask filters on last dimension to get tensor of shape (num_filtered, 4+)
    '''

    device = bboxes.device
    anchors = anchors.to(device) # In case anchors were not on device
    stride = torch.tensor(stride, device = device).flip(dims = [0]) # (stride_w, stride_h)
    fmap_h, fmap_w = bboxes.shape[-3:-1]

    if mask is not None:
        bboxes = bboxes[mask]
        
        # Shape: (num_filtered, 4), each row = (batch_idx, anchor_idx, grid_y, grid_x)
        mask_idxs = torch.nonzero(mask, as_tuple = False)
        
        grid_y = mask_idxs[:, 2]
        grid_x = mask_idxs[:, 3]
        anchors = anchors[mask_idxs[:, 1]]
    else:
        bboxes = bboxes.clone()
        
        grid_y, grid_x = torch.meshgrid(
            torch.arange(fmap_h, device = device),
            torch.arange(fmap_w, device = device),
            indexing = 'ij'
        )
        
        grid_y = grid_y.view(1, 1, fmap_h, fmap_w)
        grid_x = grid_x.view(1, 1, fmap_h, fmap_w)
        anchors = anchors.view(1, -1, 1, 1, 2) # Shape: (1, num_anchors, 1, 1, 2)
    
    # This is in CXCYWH format
    bboxes[..., 0] += grid_x # Units: fmap
    bboxes[..., 1] += grid_y # Units: fmap
    bboxes[..., 2:4] = torch.exp(bboxes[..., 2:4]) * anchors # Units: pixel

    if return_units == 'pixel':
        bboxes[..., :2] *= stride
    else:
        bboxes[..., 2:4] /= stride

    if return_format == 'xyxy':
        bboxes = convert.cxcywh_to_xyxy(bboxes)

    # Note: The decoded bboxes are in units of the input size (pixels)
    return bboxes

def decode_yolov3_targets(scale_targs: List[torch.tensor], 
                          scale_anchors: List[torch.tensor],
                          strides: List[Tuple[int, int]]):
    '''
    scale_targs (List[torch.tensor]): List of batched and encoded scale targets of shape: 
                                      (batch_size, num_anchors, fmap_h, fmap_w, 5 + C)
    scale_anchors (List[torch.tensor]): List of scale anchors corresponding to `scale_targs`.
                                        Each element has shape: (num_anchors, 2), where the last dimension gives 
                                        the (width, height) of the anchors in units of the input size (pixels).
    strides (List[Tuple[int, int]]): List of strides (height, width), one per scale associated 
                                     with `scale_targs` and `scale_anchors`.
    '''
    for i in range(len(scale_targs)):
        # This decodes all cells, but should be faster due to vectorization
        # The bboxes are decoded to XYXY format
        scale_targs[i] = decode_yolov3_bboxes(bboxes = scale_targs[i], 
                                              anchors = scale_anchors[i],
                                              stride = strides[i],
                                              return_format = 'xyxy', 
                                              return_units = 'pixel')
    targs_dicts = []
    for batch_idx in range(scale_targs[0].shape[0]):
        targs_res = {'boxes': [], 'labels': []}
        for targs in scale_targs: 
            samp_targs = targs[batch_idx] # Shape: (num_anchors, fmap_h, fmap_w, 5 + C)
            targs_obj = samp_targs[samp_targs[..., 4] == 1] # Shape: (num_objs, 5 + C)

            targs_res['boxes'].append(targs_obj[..., :4])
            targs_res['labels'].append(targs_obj[..., 5:].argmax(dim = -1))

        for key, value in targs_res.items():
            targs_res[key] = torch.concat(value, dim = 0)

        targs_dicts.append(targs_res)

    return targs_dicts
