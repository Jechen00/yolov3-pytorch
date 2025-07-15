#####################################
# Imports & Dependencies
#####################################
import torch

from typing import List, Tuple, Union, Optional, Literal, Dict
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
    Sigmoid is applied to the box offsets (tx, ty) and object confidence (to).
    Either sigmoid or softmax is applied to the class scores to get probabilities.
    
    Args:
        scale_logits (List[Tensor]): Scale logit tensors from YOLOv3, one per detecion scale.
                                     Each has shape (batch_size, num_anchors, fmap_h, fmap_w, 5 + C),
                                     where the last dimension is (tx, ty, tw, th, to, class scores).
        split_preds (bool): Whether to split outputs into bounding box, object confidence, and class components.
                            If `split_preds = True`, this function returns a tuple of 3 lists of tensors, 
                            rather than a single list of tensors.
        softmax_probs (bool): Whether to use softmax on class scores instead of sigmoid.

    Returns:
        - List[torch.Tensor] if `split_preds = False` : List of activated YOLOv3 logits, one per detection scale.
                                                        Each has shape (batch_size, num_anchors, fmap_h, fmap_w, 5 + C).
        - BBoxConfProbs if `split_preds = True`: Tuple of 3 lists of activated YOLOv3 logits. 
                                                 Each list is the same length, 
                                                 equaling the number of detection scales from `scale_logits`.
                                                 The lists are as follows:
                                                    - bbox_preds: Bounding box predictions of shape (batch_size, num_anchors, fmap_h, fmap_w, 4),
                                                                  containing (sigmoid(tx), sigmoid(ty), tw, th).
                                                    - conf_preds: Object confidence predictions of shape (batch_size, num_anchors, fmap_h, fmap_w,),
                                                                  containing sigmoid(to).
                                                    - class_preds: Class predictions of shape (batch_size, num_anchors, fmap_h, fmap_w, 5),
                                                                   where the last dimension represents class probabilities.
                                                                   If `split_preds = True`, softmax is used and these form a single distribution (sums to 1).
                                                                   If `split_preds = False`, the probabilities are independent from eachother.

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
    Decodes YOLOv3 activated logits into proper bounding box coordinates.

    Args:
        bboxes (torch.Tensor): Tensor of shape (batch_size, num_anchors, fmap_h, fmap_w, 4+),
                               where the first 4 elements of the last dimension represent
                               the encoded bounding box coordinates as (sigmoid(tx), sigmoid(ty), tw, th).
        anchors (torch.Tensor). Anchor tensor of shape (num_anchors, 2), where the last dimension gives 
                                the (width, height) of the anchors in units of the input size (pixels).
                                These anchors would correspond to the feature map scale of `bboxes`.
        stride (Tuple[int, int]): The height and width stride corresponding to the feature map scale of `bboxes`.
        mask (torch.Tensor): Boolean or binary mask tensor of shape (batch_size, num_anchors, fmap_h, fmap_w),
                             used to filter `bboxes` so that only valid entries are decoded.
        return_format ('xyxy' or 'cxcywh'): Return format of the bounding boxes coordinates after decoding.
                                                - 'xyxy': Coordinates are (xmin, ymin, xmax, ymax).
                                                - 'cxcywh': Coordinates are (xcenter, ycenter, width, height).
                                            Default is 'cxcywh'.
        return_units ('pixel' or 'fmap'): Units for decoded bounding boxes.
                                            - 'pixel': Coordinates are in units of the input size.
                                            - 'fmap': Coorinates are normalized relative to feature map, 
                                                      i.e., divides pixel coords by stride.
                                          Default is 'pixel'.

    Returns:
        decoded_bboxes (torch.Tensor): Tensor containing the decoded bounding box coordinates 
                                       in the format specified by `return_format` and the units specified by `return_units`.
                                       If `mask` is provided, shape is (num_filtered, 4+), 
                                       where the first 4 elements of the last dimension represents the bounding box coordinates.
                                       If `mask` is not provided, shape is (batch_size, num_anchors, fmap_h, fmap_w, 4+),
                                       where the first 4 elements of the last dimension represents the bounding box coordinates.
    '''
    device = bboxes.device
    anchors = anchors.to(device) # In case anchors were not on device
    stride = torch.tensor(stride, device = device).flip(dims = [0]) # (stride_w, stride_h)
    fmap_h, fmap_w = bboxes.shape[-3:-1]

    if mask is not None:
        decoded_bboxes = bboxes[mask]
        
        # Shape: (num_filtered, 4), each row = (batch_idx, anchor_idx, grid_y, grid_x)
        mask_idxs = torch.nonzero(mask, as_tuple = False)
        
        grid_y = mask_idxs[:, 2]
        grid_x = mask_idxs[:, 3]
        anchors = anchors[mask_idxs[:, 1]]
    else:
        decoded_bboxes = bboxes.clone()
        
        grid_y, grid_x = torch.meshgrid(
            torch.arange(fmap_h, device = device),
            torch.arange(fmap_w, device = device),
            indexing = 'ij'
        )
        
        grid_y = grid_y.view(1, 1, fmap_h, fmap_w)
        grid_x = grid_x.view(1, 1, fmap_h, fmap_w)
        anchors = anchors.view(1, -1, 1, 1, 2) # Shape: (1, num_anchors, 1, 1, 2)
    
    # This is in CXCYWH format
    decoded_bboxes[..., 0] += grid_x # Units: fmap
    decoded_bboxes[..., 1] += grid_y # Units: fmap
    decoded_bboxes[..., 2:4] = torch.exp(decoded_bboxes[..., 2:4]) * anchors # Units: pixel

    if return_units == 'pixel':
        decoded_bboxes[..., :2] *= stride
    else:
        decoded_bboxes[..., 2:4] /= stride

    if return_format == 'xyxy':
        decoded_bboxes = convert.cxcywh_to_xyxy(decoded_bboxes)

    # Note: The decoded bboxes are in units of the input size (pixels)
    return decoded_bboxes

def decode_yolov3_targets(scale_targs: List[torch.tensor], 
                          scale_anchors: List[torch.tensor],
                          strides: List[Tuple[int, int]]) -> List[Dict[str, torch.Tensor]]:
    '''
    Decodes YOLOv3-encoded targets into dictionaries containing class label and bounding box coordinate information.

    Args:
        scale_targs (List[torch.tensor]): List of batched YOLOv3-encoded targets, one per detection scale.
                                          Each element is a tensor of shape 
                                          (batch_size, num_anchors, fmap_h, fmap_w, 5 + C)
        scale_anchors (List[torch.tensor]): List of anchor tensors for each detection scale in `scale_targs`.
                                            Each element has shape: (num_anchors, 2), where the last dimension gives 
                                            the (width, height) of the anchor in units of the input size (pixels).
        strides (List[Tuple[int, int]]): List of strides in the form (height, width) corresponding to 
                                         each detection scale in `scale_targs`.
    Returns:
        targs_dicts (List[Dict[str, torch.Tensor]]): List of result dictionaries containing class label 
            and bounding box coordinate information for each image in the batch from `scale_targs`.

            Each result dictionary has the keys:
                - labels (torch.Tensor): Tensor of object labels. Shape is (num_objects,). 
                - boxes (torch.Tensor): Tensor of bounding box coordinates for each object in `labels`.
                                        Shape is (num_objects, 4), where the last dimension is in
                                        XYXY format (i.e. (xmin, ymin, xmax, ymax)) 
                                        and in units of the input size (pixels).
    '''
    decode_scale_targs = []
    for i in range(len(scale_targs)):
        # This decodes all cells, but should be faster due to vectorization
        # The bboxes are decoded to XYXY format
        # Shape: (batch_size, num_anchors, fmap_h, fmap_w, 5 + C)
        decode_scale_targs.append(
            decode_yolov3_bboxes(
                bboxes = scale_targs[i], 
                anchors = scale_anchors[i],
                stride = strides[i],
                return_format = 'xyxy', 
                return_units = 'pixel'
            )
        )
    targs_dicts = []
    for batch_idx in range(decode_scale_targs[0].shape[0]):
        targs_res = {'boxes': [], 'labels': []}
        for targs in decode_scale_targs:
            samp_targs = targs[batch_idx] # Shape: (num_anchors, fmap_h, fmap_w, 5 + C)
            targs_obj = samp_targs[samp_targs[..., 4] > 0] # Shape: (num_objs, 5 + C)

            targs_res['boxes'].append(targs_obj[..., :4])
            targs_res['labels'].append(targs_obj[..., 5:].argmax(dim = -1))

        for key, value in targs_res.items():
            targs_res[key] = torch.concat(value, dim = 0)

        targs_dicts.append(targs_res)

    return targs_dicts
