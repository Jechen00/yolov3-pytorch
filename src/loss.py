#####################################
# Imports & Dependencies
#####################################
import torch
from torch import nn
import torch.nn.functional as F

from typing import List, Optional, Dict, Literal

from src import postprocess, evaluate


#####################################
# Loss Class
#####################################
class YOLOv3Loss(nn.Module):
    def __init__(self,
                 lambda_coord: float = 1.0, 
                 lambda_class: float = 1.0, 
                 lambda_conf: float = 1.0,
                 alpha: float = 0.25,
                 gamma: float = 2.0,
                 class_smoothing: float = 0.0,
                 use_iou_coord: bool = False,
                 softmax_probs: bool = False,
                 iou_coord_reg: Optional[Literal['giou', 'diou', 'ciou']] = None,
                 scale_weights: Optional[List[float]] = None,
                 scale_anchors: Optional[List[torch.Tensor]] = None,
                 strides = None):
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
        self.use_iou_coord = use_iou_coord
        self.softmax_probs = softmax_probs
        self.iou_coord_reg = iou_coord_reg
        self.scale_anchors = scale_anchors
        self.strides = strides

        if scale_weights is not None:
            self.scale_weights = scale_weights
        else:
            self.scale_weights = [1, 1, 1]

    def forward(
        self,
        scale_logits: List[torch.Tensor],
        scale_targs: List[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        loss_dict = {key: 0.0 for key in self.loss_keys}

        # Loop over scales and sum their total losses
        for i, (targs, logits) in enumerate(zip(scale_targs, scale_logits)):
            assert not torch.isnan(logits).any(), 'NaNs in `scale_logits`'
            assert not torch.isnan(targs).any(), 'NaNs in `scale_targs`'

            # Loss components are defaulted to 0
            loss_comps = {key: 0.0 for key in self.loss_keys[:-1]}

            obj_mask = (targs[..., 4] == 1)
            valid_mask = torch.logical_not(targs[..., 4] == -1)

            #----------------------------
            # Confidence Loss (Focal)
            #----------------------------
            conf_logits = logits[valid_mask][..., 4]
            conf_targs = targs[valid_mask][..., 4]
            
            bce_conf_losses = F.binary_cross_entropy_with_logits(
                input = conf_logits,
                target = conf_targs,
                reduction = 'none'
            )

            # targ = 1 --> p_t = prob; alpha_t = alpha
            # targ = 0 --> p_t = 1 - prob; alpha_t = 1 - alpha
            conf_preds = torch.sigmoid(conf_logits)
            p_t = torch.where(conf_targs == 1, conf_preds, 1 - conf_preds)
            alpha_t = torch.where(conf_targs == 1, self.alpha, 1 - self.alpha)

            conf_focal_loss = alpha_t * (1 - p_t)**self.gamma * bce_conf_losses
            loss_comps['conf'] = conf_focal_loss.sum()

            # Compute object component losses if objects exist
            if obj_mask.any():
                targs_obj = targs[obj_mask] # Shape: (num_objects, 5 + C)

                #----------------------------
                # Class Loss
                #----------------------------
                if not self.softmax_probs:
                    # Binary cross-entropy on each class (multi-label, default YOLOv3 behavior)
                    loss_comps['class'] = F.binary_cross_entropy_with_logits(
                        input = logits[obj_mask][:, 5:],
                        target = targs_obj[:, 5:],
                        reduction = 'sum'
                    )
                    
                else:
                    # Cross-entropy with logits and class indices (single-label)
                    loss_comps['class'] =  F.cross_entropy(
                        input = logits[obj_mask][:, 5:],
                        target = targs_obj[:, 5:].argmax(dim = -1),
                        reduction = 'sum',
                        label_smoothing = self.class_smoothing
                    )

                #----------------------------
                # Coordinate Loss
                #----------------------------
                # Apply sigmoid to t_x, t_y (center offsets)
                preds = logits.clone()
                preds[..., :2] = torch.sigmoid(preds[..., :2])

                if not self.use_iou_coord:
                    loss_comps['coord'] = F.mse_loss(
                        input = preds[obj_mask][:, :4],
                        target = targs_obj[:, :4],
                        reduction = 'sum'
                    )

                else:
                    preds_bboxes = postprocess.decode_yolov3_bboxes(
                        bboxes = preds[..., :4], 
                        anchors = self.scale_anchors[i],
                        stride = self.strides[i],
                        mask = obj_mask, 
                        return_format = 'xyxy',
                        return_units = 'pixel'
                    )
                    targs_bboxes = postprocess.decode_yolov3_bboxes(
                        bboxes = targs[..., :4], 
                        anchors = self.scale_anchors[i],
                        stride = self.strides[i],
                        mask = obj_mask, 
                        return_format = 'xyxy',
                        return_units = 'pixel'
                    )
                    
                    _, reg_ious = evaluate.calc_ious(bboxes1 = preds_bboxes, bboxes2 = targs_bboxes, 
                                                     elementwise = True, reg_type = self.iou_coord_reg)
                    
                    loss_comps['coord'] = (1 - reg_ious).sum()

            #----------------------------
            # Full YOLOv3 Loss
            #----------------------------
            # Add each component loss to loss_dict
            for key, value in loss_comps.items():
                loss_dict[key] += value

            loss_dict['total'] += self.scale_weights[i] * sum(
                self.lambdas[key] * loss_comps[key] 
                for key in loss_comps.keys()
            )

        # Divide by batch size to get mean across batch samples
        batch_size = scale_targs[0].shape[0]
        for key in loss_dict:
            loss_dict[key] = loss_dict[key] / batch_size
            
        return loss_dict
