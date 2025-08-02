#####################################
# Imports & Dependencies
#####################################
import torch
from torch import nn
import torch.nn.functional as F

from typing import List, Optional, Dict, Literal, Tuple

from src import postprocess, evaluate


#####################################
# Loss Class
#####################################
class YOLOv3Loss(nn.Module):
    '''
    Loss function for training a YOLOv3 model.

    Args:
        lambda_coord (float): Weight of the localization loss. Default is 1.0.
        lambda_class (float): Weight of the classification loss. Default is 1.0.
        lambda_conf (float): Weight of the object confidence loss. Default is 1.0.
        use_focal_conf (bool): Whether to use focal loss for object confidence, 
                               as is done in https://arxiv.org/pdf/1708.02002.
                               If False, the standard binary cross-entropy (BCE) loss is used.
                               Default is False.
        use_iou_coord (bool): Whether to use an IoU-based loss (IoU, GIoU, DIoU, or CIoU) for localization,
                              as is done in https://arxiv.org/pdf/1608.01471.
                              If False, the standard mean squared error (MSE) loss is used.
                              If True, the following are required parameters: `scale_anchors`, `strides`.
                              Default is False.
        softmax_probs (bool): Whether to softmax class probabilities and use cross-entropy (CE) loss for classification.
                              If False, sigmoid is applied to each class prediction and the standard BCE loss is used
                              (treats each class as independent logisitic regression tasks).
                              Default is False.
        alpha (float): The alpha-parameter in focal loss used to balance importance between positive and negative examples. Default is 0.25.
        gamma (float): The gamma-parameter in focal loss used to control how much to down-weight easy examples. Default is 2.0.
        class_smoothing (float): Amount of label smoothing to apply to classification targets. Default is 0.0 (no smoothing).
        iou_coord_reg (optional, Literal['giou', 'diou', 'ciou']): The type of regularization term to use if `use_iou_coord = True`.
                                                                   If not provided, regular IoU loss is used.
                                                                   When provided, the respective GIoU, DIoU, or CIoU regularizations are used.
                                                                   Default is None.
        ignore_threshold (optional, float): IoU threshold for the ignore mask used for the objectness loss.
                                            If a bounding box prediction in a no-object cell (target objectess = 0) 
                                            has an IoU greater than or equal to `ignore_threshold`
                                            with any target (ground truth) bounding box in an object cell (target objectness > 0),
                                            that no-object cell will be excluded from the objectness loss computation.
                                            **Note:** This differs from the ignore threshold used in YOLOv3 target encoding/assignment, 
                                            where a grid cell is marked as "ignore" (target objectness = -1) based on the IoU between 
                                            the target bounding box and anchor boxes (priors) with the same center coordinate (grid_y, grid_x).
                                            If `ignore_threshold` is None, no ignore mask is applied during loss computations.
                                            Default is None.
        scale_weights (optional, List[float]): List of weights to apply to the loss at each scale of the model.
                                               The length of this list must match the number of scales in the model (e.g., [large, medium, small]). 
                                               Each scale's loss is multiplied by its corresponding weight:  
                                                    `total_loss = large_weight * large_loss + medium_weight * medium_loss + small_weight * small_loss`.
                                               If not provided, all scales are weighted equally. Default is None.
        scale_anchors (optional, List[torch.tensor]): List of anchor tensors for each scale of the model.
                                                      Each element has shape: (num_anchors, 2), where the last dimension gives 
                                                      the (width, height) of the anchor in units of the input size (pixels). 
                                                      Default is None.
        strides (optional, List[Tuple[int, int]]): List of strides (height, width) corresponding to each scale of the model.
                                                   Default is None.
    '''
    def __init__(self,
                 lambda_coord: float = 1.0, 
                 lambda_class: float = 1.0, 
                 lambda_conf: float = 1.0,
                 use_focal_conf: bool = False,
                 use_iou_coord: bool = False,
                 softmax_probs: bool = False,
                 alpha: float = 0.25,
                 gamma: float = 2.0,
                 class_smoothing: float = 0.0,
                 iou_coord_reg: Optional[Literal['giou', 'diou', 'ciou']] = None,
                 ignore_threshold: Optional[float] = None,
                 scale_weights: Optional[List[float]] = None,
                 scale_anchors: Optional[List[torch.Tensor]] = None,
                 strides: Optional[List[Tuple[int, int]]] = None):
        super().__init__()
        if use_iou_coord:
            assert (scale_anchors is not None) and (strides is not None), (
                'If `use_iou_coord = True`, `scale_anchors` and `strides` must be provided.'
            )

        self.lambdas = {
            'coord': lambda_coord,
            'class': lambda_class,
            'conf': lambda_conf
        }
        self.loss_keys = list(self.lambdas.keys()) + ['total']

        self.alpha = alpha
        self.gamma = gamma
        self.class_smoothing = class_smoothing
        self.use_focal_conf = use_focal_conf
        self.use_iou_coord = use_iou_coord
        self.softmax_probs = softmax_probs
        self.iou_coord_reg = iou_coord_reg
        self.ignore_threshold = ignore_threshold
        self.scale_weights = scale_weights
        self.scale_anchors = scale_anchors
        self.strides = strides

    def create_ignore_mask(
        self,
        logits: torch.Tensor, 
        targs: torch.Tensor, 
        anchors: torch.Tensor,
        stride: Tuple[int, int],
        return_targs_bboxes: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        '''
        Creates an ignore mask for objectness loss computation.

        If a bounding box prediction in a no-object cell (target objectess = 0) 
        has an IoU greater than or equal to `self.ignore_threshold`
        with any target (ground truth) bounding box in an object cell (target objectness > 0),
        the no-object cell will be marked as `True` in the ignore mask. 

        Ignored cells are excluded from the objectness loss computation to prevent 
        penalizing reasonable predictions.

        **Note:** This differs from the ignore process in YOLOv3 target encoding/assignment, 
        where a grid cell is marked as "ignore" (target objectness = -1) based on the IoU between 
        the target bounding box and anchor boxes (priors) with the same center coordinate (grid_y, grid_x).

        Args:
            logits (torch.Tensor): Logits from a single detection scales of a YOLOv3 model output.
                                   Shape is (batch_size, num_anchors, fmap_h, fmap_w, 5 + C),
                                   where the last dimension represents (tx, ty, tw, th, to, class scores).
            targs: (torch.Tensor): YOLOv3-encoded targets (ground truths) assigned to the same scale as `logits`.
                                   Shape is the same as `logits` and the format should be the same
                                   as the model outputs **after activation**,
                                   i.e., after applying sigmoid to box coordinates (tx, ty) and object confidence (to),
                                   and sigmoid or softmax to class scores.
            anchors (torch.Tensor): Anchor tensors for the same scale as `logits`.
                                    Shape is (num_anchors, 2), where the last dimension gives 
                                    the (width, height) of the anchor in units of the input size (pixels). 
            stride(Tuple[int, int]): The stride (height, width) for the same scale as `logits`.
            return_targs_bboxes (bool): Whether to return the decoded target (ground truth) bounding boxes from object cells.
                                        Default is False.

        Returns:
            ignore_mask (torch.Tensor): Boolean mask indicating which cells, 
                                        i.e. coordinates represented by (anchor index, grid_y, grid_x),
                                        should be ignored during objectness loss computation.
                                        Shape is (batch_size, num_anchors, grid_y, grid_x)
            obj_targs_bboxes (optional, torch.Tensor): Decoded target (ground truth) bounding boxes.
                                                       Shape is (num_objects, 4), where the last dimension 
                                                       is in XYXY format and are in units of the input image (pixels).
                                                       Only returned if `return_targs_bboxes=True`; otherwise, returns None.

        '''
        # Shape of noobj_idxs: (num_noobj, 4) where 4 = (batch_idx, anchor_idx, grid_y, grid_x)
        noobj_mask = (targs[..., 4] == 0)
        noobj_idxs = noobj_mask.nonzero(as_tuple = False)
        ignore_mask = torch.zeros_like(noobj_mask)

        # Shape of obj_idxs: (num_obj, 4) where 4 = (batch_idx, anchor_idx, grid_y, grid_x)
        obj_mask = (targs[..., 4] > 0)
        obj_idxs = obj_mask.nonzero(as_tuple = False)

        logits_bboxes = logits[..., :4].clone()
        logits_bboxes[..., :2] = torch.sigmoid(logits_bboxes[..., :2]) # Apply sigmoid to t_x, t_y (center offsets)

        noobj_preds_bboxes = postprocess.decode_yolov3_bboxes(
            bboxes = logits_bboxes,
            anchors = anchors,
            stride = stride,
            mask = noobj_mask,
            return_format = 'xyxy',
            return_units = 'pixel'
        )

        obj_targs_bboxes = postprocess.decode_yolov3_bboxes(
            bboxes = targs[..., :4],
            anchors = anchors,
            stride = stride,
            mask = obj_mask,
            return_format = 'xyxy',
            return_units = 'pixel'
        )

        for b in range(targs.shape[0]):
            noobj_b_mask = (noobj_idxs[:, 0] == b)
            obj_b_mask = (obj_idxs[:, 0] == b)

            if torch.any(obj_b_mask):
                ious = evaluate.calc_ious(bboxes1 = noobj_preds_bboxes[noobj_b_mask],
                                          bboxes2 = obj_targs_bboxes[obj_b_mask])
                max_ious, _ = ious.max(dim = -1)

                # Shape: (num_ignore, 3), where 3 = (anchor_idx, grid_y, grid_x)
                ignore_idxs = noobj_idxs[noobj_b_mask][max_ious >= self.ignore_threshold][:, 1:]
                ignore_mask[b, ignore_idxs[:, 0], ignore_idxs[:, 1], ignore_idxs[:, 2]] = 1
                
        if return_targs_bboxes:
            return ignore_mask, obj_targs_bboxes
        else:
            return ignore_mask, None
    
    def forward(self,
                scale_logits: List[torch.Tensor],
                scale_targs: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        '''
        Computes the YOLOv3 loss across a batch. Returns a dictionary containing the total loss and its unweighted components.

        Args:
            scale_logits (List[torch.Tensor]): List of logits from the YOLOv3 model, one per detection scale.
                                               Each element is a tensor of shape: (batch_size, num_anchors, fmap_h, fmap_w, 5 + C),
                                               where the last dimension represents (tx, ty, tw, th, to, class scores).
            scale_targs: (List[torch.Tensor]): List of YOLOv3-encoded targets (ground truths), one per detection scale.
                                               Each element is a tensor with the same shape as `scale_logits`.
                                               Each target should match the format of the model outputs **after activation**,
                                               i.e., after applying sigmoid to box coordinates and object confidence,
                                               and sigmoid or softmax to class scores.
        Returns:
            loss_dict (Dict[str, torch.Tensor]): Dictionary containing the loss component values (scalar tensors).
                                                 The keys are as follows:
                                                    - total: The total loss, summing together all weighted components of the YOLOv3 loss.
                                                    - coord: The unweighted coordinate/localization loss from object cells (target P(object) > 0).
                                                    - class: The unweighted classification loss from object cells (target P(object) > 0).
                                                    - conf: The unweighted object confidence loss from all valid cells (target P(object) != -1).
        '''
        loss_dict = {key: 0.0 for key in self.loss_keys}
        num_classes = scale_targs[0].shape[-1] - 5

        if self.scale_weights is None:
            scale_weights = [1] * len(scale_targs) # All scales are treated equally
        else:
            scale_weights = self.scale_weights

        # Loop over scales and sum their total losses
        for i, (targs, logits) in enumerate(zip(scale_targs, scale_logits)):
            assert not torch.isnan(logits).any(), 'NaNs in `scale_logits`'
            assert not torch.isnan(targs).any(), 'NaNs in `scale_targs`'

            # Loss components are defaulted to 0
            loss_comps = {key: 0.0 for key in self.loss_keys[:-1]}

            obj_mask = (targs[..., 4] > 0)
            if self.ignore_threshold is None:
                # Only ignore no-object cells based on anchor assignment from the dataset
                valid_mask = (targs[..., 4] != -1)
                targs_bboxes = None # Placeholder in case use_iou_coord = True

            else:
                # Ignore no-object cells if marked as ignored by dataset anchor assignment 
                    # or if prediction overlaps too much with a GT bbox
                ignore_mask, targs_bboxes = self.create_ignore_mask(
                    logits = logits, 
                    targs = targs,
                    anchors = self.scale_anchors[i],
                    stride = self.strides[i],
                    return_targs_bboxes = self.use_iou_coord
                )
                valid_mask = torch.logical_and((targs[..., 4] != -1), torch.logical_not(ignore_mask))

            #------------------
            # Confidence Loss
            #------------------
            conf_logits = logits[valid_mask][..., 4]
            conf_targs = targs[valid_mask][..., 4]
            
            bce_conf_losses = F.binary_cross_entropy_with_logits(
                input = conf_logits,
                target = conf_targs,
                reduction = 'none'
            )

            if not self.use_focal_conf:
                # Possibly add pos_weight or separate obj and nobj for this?
                loss_comps['conf'] = bce_conf_losses.sum()

            else:
                # targ > 0 --> p_t = prob; alpha_t = alpha
                # targ = 0 --> p_t = 1 - prob; alpha_t = 1 - alpha
                conf_preds = torch.sigmoid(conf_logits)
                p_t = torch.where(conf_targs > 0, conf_preds, 1 - conf_preds)
                alpha_t = torch.where(conf_targs > 0, self.alpha, 1 - self.alpha)

                conf_focal_loss = alpha_t * (1 - p_t)**self.gamma * bce_conf_losses
                loss_comps['conf'] = conf_focal_loss.sum()

            # Compute object component losses if objects exist
            if obj_mask.any():
                targs_obj = targs[obj_mask] # Shape: (num_objects, 5 + C)
                obj_weights = targs_obj[..., 4] # Shape: (num_objects,) and represents P(object)

                #------------------
                # Class Loss
                #------------------
                if not self.softmax_probs:
                    # Apply optional label smoothing
                    if self.class_smoothing > 0:
                        targs_probs = targs_obj[:, 5:] * (1 - self.class_smoothing) + (self.class_smoothing / num_classes)
                    else:
                        targs_probs = targs_obj[:, 5:]

                    # Binary cross-entropy on each class (multi-label, default YOLOv3 behavior)
                    class_loss = F.binary_cross_entropy_with_logits(
                        input = logits[obj_mask][:, 5:],
                        target = targs_probs,
                        reduction = 'none'
                    ).sum(dim = -1)
                    
                else:
                    # Cross-entropy with logits and class indices (single-label)
                    class_loss =  F.cross_entropy(
                        input = logits[obj_mask][:, 5:],
                        target = targs_obj[:, 5:].argmax(dim = -1),
                        reduction = 'none',
                        label_smoothing = self.class_smoothing
                    )

                # class_loss shape: (num_objects,)
                loss_comps['class'] = (obj_weights * class_loss).sum()

                #------------------
                # Coordinate Loss
                #------------------
                # Apply sigmoid to t_x, t_y (center offsets)
                preds = logits.clone()
                preds[..., :2] = torch.sigmoid(preds[..., :2])

                if not self.use_iou_coord:
                    coord_loss = F.mse_loss(
                        input = preds[obj_mask][:, :4],
                        target = targs_obj[:, :4],
                        reduction = 'none'
                    ).sum(dim = -1)

                else:
                    if targs_bboxes is None:
                        targs_bboxes = postprocess.decode_yolov3_bboxes(
                            bboxes = targs[..., :4],
                            anchors = self.scale_anchors[i],
                            stride = self.strides[i],
                            mask = obj_mask,
                            return_format = 'xyxy',
                            return_units = 'pixel'
                        )

                    preds_bboxes = postprocess.decode_yolov3_bboxes(
                        bboxes = preds[..., :4], 
                        anchors = self.scale_anchors[i],
                        stride = self.strides[i],
                        mask = obj_mask, 
                        return_format = 'xyxy',
                        return_units = 'pixel'
                    )
                    
                    _, reg_ious = evaluate.calc_ious(bboxes1 = preds_bboxes, bboxes2 = targs_bboxes, 
                                                     elementwise = True, reg_type = self.iou_coord_reg)
                    coord_loss = 1 - reg_ious

                # coord_loss shape: (num_objects,)
                loss_comps['coord'] = (obj_weights * coord_loss).sum()

            #------------------
            # Full YOLOv3 Loss
            #------------------
            # Add each component loss to loss_dict
            for key, value in loss_comps.items():
                loss_dict[key] += value
            
            loss_dict['total'] += scale_weights[i] * sum(
                self.lambdas[key] * loss_comps[key] 
                for key in loss_comps.keys()
            )

        # Divide by batch size to get mean across batch samples
        batch_size = scale_targs[0].shape[0]
        for key in loss_dict:
            loss_dict[key] = loss_dict[key] / batch_size
            
        return loss_dict