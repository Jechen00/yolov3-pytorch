#####################################
# Functions
#####################################
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.ops import batched_nms
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from typing import List, Optional, Union, Tuple, Literal, Dict, Any

from src import postprocess
from src.utils import convert


#####################################
# Functions
#####################################
# ----------------------------
# Intersection Over Union
# ----------------------------
def bbox_area(bboxes: torch.Tensor) -> torch.Tensor:
    '''
    Computes the area of bounding boxes.
    
    Args:
        bboxes (torch.Tensor): Tensor of shape (..., 4+),
                               where the last dimension represents bounding boxes in the format:
                               (x_min, y_min, x_max, y_max).

    Returns:
        torch.Tensor: Tensor of shape (..., ) containing
                      the area of each bounding box.
    '''
    
    # Clamp assigns an area of 0 to malformed bboxes
    bbox_widths = (bboxes[..., 2] - bboxes[..., 0]).clamp(min = 0) # Width is x_max - x_min
    bbox_heights = (bboxes[..., 3] - bboxes[..., 1]).clamp(min = 0) # Height is y_max - y_min
    return bbox_widths * bbox_heights

def calc_ious_wh(bbox_whs_1: torch.Tensor, 
                 bbox_whs_2: torch.Tensor, 
                 elementwise: bool = False) -> torch.Tensor:
    '''
    Computes Intersection over Union (IoU) between two set of bounding boxes,
    given only their widths and heights (i.e. assumes the bounding boxes have the same centers).
    
    Args:
        bbox_whs_1 (torch.Tensor): Tensor of shape (..., num_bboxes1, 2)
                                   containing the first set of bounding box widths and heights.
        bbox_whs_2 (torch.Tensor): Tensor of shape (..., num_bboxes2, 2)
                                   containing the second set of bounding box widths and heights.
        elementwise (bool): Whether to calculate IoUs element-wise (1:1), or pairwise (all combinations).
                            If True, `bbox_whs_1` and `bbox_whs_2` must have the same shape.
        
    Returns:
        ious (torch.Tensor): IoU values. If `elementwise = False`, shape is (..., num_bboxes1, num_bboxes2), 
                             where the entry at [i, j] gives the IoU between the i-th box in `bbox_whs_1`
                             and j-th box in `bbox_whs_2`. If `elementwise = True`, shape is (..., num_bboxes1),
                            where the i-th entry gives the IoU between the i-th boxes of `bbox_whs_1` and` bbox_whs_2`.
    '''
    if elementwise:
        assert bbox_whs_1.shape == bbox_whs_2.shape, (
            'If `elementwise = True`, `bbox_whs_1` and `bbox_whs_2` must have the same shape.'
        )
        
        bbox_whs_1 = bbox_whs_1[..., :2] # Shape: (..., num_bboxes1, 2)
        bbox_whs_2 = bbox_whs_2[..., :2] # Shape: (..., num_bboxes2, 2)
    else:
        bbox_whs_1 = bbox_whs_1[..., :, None, :2] # Shape: (..., num_bboxes1, 1, 2)
        bbox_whs_2 = bbox_whs_2[..., None, :, :2] # Shape: (..., 1, num_bboxes2, 2)

    inter_areas = torch.min(bbox_whs_1, bbox_whs_2).clamp(min = 0).prod(dim = -1)
    union_areas = bbox_whs_1.prod(dim = -1) + bbox_whs_2.prod(dim = -1) - inter_areas

    return inter_areas / (union_areas + 1e-7) # Shape: (num_bboxes1, num_bboxes_2)

def calc_ious(
    bboxes1: torch.Tensor, 
    bboxes2: torch.Tensor, 
    elementwise: bool = False,
    reg_type: Optional[Literal['giou', 'diou', 'ciou']] = None
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    '''
    Computes Intersection over Union (IoU) between two set of bounding boxes,
    with the option of using a regularization term for Generalized IoU (GIoU) or Complete IoU (CIoU).
    
    The range of IoU is [0, 1].
    The range of GIoU, DIoU, and CIoU is [-1, 1].
    
    Args:
        bboxes1 (torch.Tensor): Tensor of shape (..., num_bboxes1, 4+)
                                containing the first set of bounding boxes in 
                                (x_min, y_min, x_max, y_max) format as the first 4 elements.
        bboxes2 (torch.Tensor): Tensor of shape (..., num_bboxes2, 4+)
                                containing the second set of bounding boxes in 
                                (x_min, y_min, x_max, y_max) format as the first 4 elements.
        elementwise (bool): Whether to calculate IoUs element-wise (1:1), or pairwise (all combinations).
                            If True, `bboxes1` and `bboxes2` must have the same shape.
        reg_type (optional, Literal['giou', 'diou', 'ciou']): The type of regularization term to use.
                                    If not provided, no regularization is used and only IoU is returned.
                                    When provided, the respective GIoU, DIoU, or CIoU regularizations are used.
                                    Additionally, the IoU values are returned alongside the regularized values.
        
    Returns:
        ious (torch.Tensor): IoU values. If `elementwise = False`, shape is (..., num_bboxes1, num_bboxes2), 
                      where the entry at [i, j] gives the IoU between the i-th box in `bboxes1`
                      and j-th box in `bboxes2`. If `elementwise = True`, shape is (..., num_bboxes1),
                      where the i-th entry gives the IoU between the i-th boxes of `bboxes1` and` bboxes2`.
                      
        reg_ious (torch.Tensor): GIoU, DIoU, or CIoU values. The format of this tensor is the same as `ious`. 
                                 This is only returned along with `ious` if `reg_type` is provided.
    '''
    eps = 1e-7 # Used to prevent divide by zero errors
    
    if elementwise:
        assert bboxes2.shape == bboxes1.shape, (
            'If `elementwise = True`, `bboxes1` and `bboxes2` must have the same shape.'
        )
        
        bboxes1 = bboxes1[..., :4] # Shape: (..., num_bboxes1, 4)
        bboxes2 = bboxes2[..., :4] # Shape: (..., num_bboxes2, 4)
    else:
        bboxes1 = bboxes1[..., :, None, :4] # Shape: (..., num_bboxes1, 1, 4)
        bboxes2 = bboxes2[..., None, :, :4] # Shape: (..., 1, num_bboxes2, 4)
       
    # Upper-left and bottom-right corners of intersections
    inter_ul = torch.max(bboxes2[..., :2], bboxes1[..., :2])
    inter_br = torch.min(bboxes2[..., 2:4], bboxes1[..., 2:4])
    
    # Intersection width and height -> intersection area
    # Clamp to 0 avoid negative values and indicates no overlap
    inter_areas = (inter_br - inter_ul).clamp(min = 0).prod(dim = -1)
    
    # Union areas: area(A) + area(B) - area(intersection)
    union_areas = bbox_area(bboxes2) + bbox_area(bboxes1) - inter_areas
    
    ious = inter_areas / (union_areas + eps)
    
    if reg_type is None:
        return ious
    
    else:
        # Upper-left and bottom-right corners of minimum enclosing box
        enclose_ul = torch.min(bboxes1[..., :2], bboxes2[..., :2])
        enclose_br = torch.max(bboxes1[..., 2:4], bboxes2[..., 2:4])
        enclose_wh = (enclose_br - enclose_ul).clamp(min = 0)
        
        if reg_type == 'giou':
            enclose_areas = enclose_wh.prod(dim = -1)
            penalties = [(enclose_areas - union_areas) / (enclose_areas + eps)] # GIoU regularization term
            
        elif reg_type in ['diou', 'ciou']:
            penalties = []
            # -----------------------------
            # Center Regularization
            # -----------------------------
            bboxes2_cxcywh = convert.xyxy_to_cxcywh(bboxes2)
            bboxes1_cxcywh = convert.xyxy_to_cxcywh(bboxes1)

            # Squared diagonal of minimum enclosing box
            c2 = enclose_wh.pow(2).sum(dim = -1)

            # Squared distance between the centers of bboxes1 and bboxes2
            rho2 = (bboxes1_cxcywh[..., :2] - bboxes2_cxcywh[..., :2]).pow(2).sum(dim = -1)

            penalties.append(rho2 / (c2 + eps)) # Center distance penalty term

            if reg_type == 'ciou':
                # -----------------------------
                # Aspect Ratio Regularization
                # -----------------------------
                # Bbox aspect ratios
                bboxes1_ar = bboxes1_cxcywh[..., 2] / (bboxes1_cxcywh[..., 3] + eps)
                bboxes2_ar = bboxes2_cxcywh[..., 2] / (bboxes2_cxcywh[..., 3] + eps)

                v = 4 / torch.pi**2 * (bboxes1_ar.arctan() - bboxes2_ar.arctan()).pow(2)

                with torch.no_grad():
                    # When IoU < 0.5, aspect ratio isn't important and aren't penalized
                    alpha = torch.where(ious < 0.5, 0, v/(1 - ious + v + eps))

                penalties.append(alpha * v) # Aspect ratio penalty term

        reg_ious = ious - torch.stack(penalties, dim = 0).sum(dim = 0)
        return ious, reg_ious


# ------------------------
# Mean Average Precision
# ------------------------
def calc_dataset_map(model: nn.Module, 
                     dataloader: DataLoader, 
                     scale_anchors: List[torch.Tensor],
                     strides: List[Tuple[int, int]],
                     obj_threshold: float = 0.25,
                     nms_threshold: float = 0.5,
                     map_thresholds: Optional[List[float]] = None,
                     softmax_probs: bool = False,
                     device: Union[torch.device, str] = 'cpu',
                     **kwargs) -> dict:
    '''
    Evaluates a YOLOv3 model on a given dataset using the MeanAveragePrecision metric
    from `torchmetrics.detection`. 
    This includes the metrics: mean Average Precision (mAP) and mean Average Recall (mAR).

    Args:
        model (nn.Module): An instance of a YOLOv3 model to predict bounding boxes and class labels.
                           This model should already be moved to the specified `device`.
        dataloader (data.Dataloader): The dataloader used to transform and load the dataset in batches.
        scale_anchors (List[torch.tensor]): List of anchor tensors for each output scale of the model.
                                            Each element has shape: (num_anchors, 2), where the last dimension gives 
                                            the (width, height) of the anchor in units of the input size (pixels).
        strides (List[Tuple[int, int]]): List of strides (height, width) corresponding to each scale in `scale_anchors`.
        obj_threshold (float): Threshold to filter out low predicted object confidence scores. Default is 0.25.
        nms_threshold (float): The IoU threshold used when performing Non-Maximum Suppression (NMS). Default is 0.5.
        map_thresholds (optional, List[float]): A list of IoU thresholds used for mAP calculations.
                                                If not provided, this defaults to [0.5].
        softmax_probs (bool): Whether to softmax class logits instead of using sigmoid when predicting. Default is False.
        device (torch.device or str): The device to perform calculations on. Default is 'cpu'.
        **kwargs: Any other arguments for the `torchmetrics.detection.mean_ap.MeanAveragePrecision` class.
                  Note that the following arguments are overwritten: 
                  `box_format='xyxy'`, 'iou_thresholds=map_thresholds`.
    
    Returns:
        dict: Dictionary containing mAP and mean Average Recall (mAR) metrics as computed by
              `torchmetrics.detection.mean_ap.MeanAveragePrecision`.

              The most important keys are:
                - map (torch.Tensor): The overall mAP value calculated across all classes and IoU thresholds.
                - map_per_class (torch.Tensor): List of per-class AP values across all IoU thresholds.

              For more information, see:
                https://lightning.ai/docs/torchmetrics/stable/detection/mean_average_precision.html
    '''
    kwargs['box_format'] = 'xyxy'
    kwargs['iou_thresholds'] = [0.5] if map_thresholds is None else map_thresholds

    map_metric = MeanAveragePrecision(**kwargs)
    model.eval()
    for imgs, scale_targs in dataloader:
        scale_targs = [targs.to(device) for targs in scale_targs]
        imgs = imgs.to(device)

        targs_dicts = postprocess.decode_yolov3_targets(
            scale_targs = scale_targs, 
            scale_anchors = scale_anchors,
            strides = strides
        )
        
        # Already uses inference mode
        preds_dicts = predict_yolov3(
            model = model, 
            X = imgs, 
            scale_anchors = scale_anchors, 
            strides = strides,
            obj_threshold = obj_threshold, 
            nms_threshold = nms_threshold, 
            softmax_probs = softmax_probs
        )

        map_metric.update(preds_dicts, targs_dicts)
        
    return map_metric.compute()


# ------------------
# Prediction
# ------------------
def predict_yolov3(model: nn.Module,
                   X: torch.Tensor,
                   scale_anchors: List[torch.Tensor],
                   strides: List[Tuple[int, int]],
                   obj_threshold: float = 0.2,
                   nms_threshold: float = 0.5,
                   softmax_probs: bool = False) -> List[Dict[str, Any]]:
    '''
    This is a wrapper function for `predict_yolov3_from_logits`. 
    It uses a batch of preprocessed images `X` as input, rather than output logits from the model.

    Args:
        model (nn.Module): The YOLOv3 model to predict with. Will be set to `.eval()` mode if not done so already.
        X (torch.Tensor): The batch of preprocessed images to predict on. This should be on the same device as `model`.
                          Shape is (batch_size, channels, height, width). 
        scale_anchors (List[torch.tensor]): List of anchor tensors for each detection scale of `model`.
                                            Each element has shape: (num_anchors, 2), where the last dimension gives 
                                            the (width, height) of the anchor in units of the input size (pixels).
        strides (List[Tuple[int, int]]): List of strides (height, width) corresponding to each scale in `model`.
        obj_threshold (float): The probability threshold to filter out low predicted object confidences. Default is 0.2.
        nms_threshold (float): The IoU threshold used when performing NMS. Default is 0.5.
        softmax_probs (bool): Whether to use softmax on class scores instead of sigmoid.

    Returns:
        pred_dicts (List[dict]): A list containing prediction dictionaries for each image sample in X.
                                 For more details, see `predict_yolov3_from_logits`.                  
    '''
    assert len(X.shape) == 4, (
        'Incorrect number of dimensions for `X`. Expecting shape (batch_size, channels, height, width).'
    )

    # Set to evaluation mode if model was previously in .train()
    if model.training:
        model.eval()

    with torch.inference_mode():
        scale_logits = model(X) # List of tensors with shape: (batch_size, num_anchors, fmap_h, fmap_w, 5 + C)
    
    preds_dicts = predict_yolov3_from_logits(
        scale_logits = scale_logits,
        scale_anchors = scale_anchors,
        strides = strides,
        obj_threshold  = obj_threshold, 
        nms_threshold = nms_threshold,
        softmax_probs = softmax_probs
    )
    return preds_dicts

def predict_yolov3_from_logits(scale_logits: List[torch.Tensor], 
                               scale_anchors: List[torch.Tensor],
                               strides: List[Tuple[int, int]],
                               obj_threshold: float = 0.25,
                               nms_threshold: float = 0.5,
                               activate_logits: bool = True,
                               softmax_probs: bool = False) -> List[Dict[str, Any]]:
    '''
    Uses YOLOv3 output logits to predict the bounding boxes and class labels for a batch of preprocessed images.
    The predictions are first filtered by object confidence and then filtered by Non-Maximum Suppression (NMS).
    The bounding box predictions are returned in (x_min, y_min, x_max, y_max) format.

    Args:
        scale_logits (List[torch.Tensor]): List of logits from the YOLOv3 model, one per detection scale.
                                           Each element is a tensor of shape: (batch_size, num_anchors, fmap_h, fmap_w, 5 + C),
                                           where the last dimension represents (tx, ty, tw, th, to, class scores).
                                           All tensors in `scale_logits` should be on the same device.
        scale_anchors (List[torch.tensor]): List of anchor tensors for each detection scale of the model.
                                            Each element has shape: (num_anchors, 2), where the last dimension gives 
                                            the (width, height) of the anchor in units of the input size (pixels).
        strides (List[Tuple[int, int]]): List of strides (height, width) corresponding to each scale in 
                                         `scale_logits` and `scale_anchors`.
        obj_threshold (float): The probability threshold to filter out low predicted object confidences. Default is 0.2.
        nms_threshold (float): The IoU threshold used when performing NMS. Default is 0.5.
        activate_logits (bool): Whether logits should be activated with sigmoid (and optionally softmax) before decoding. 
                                If `activate_logits = False`, it is assumed that `scale_logits` is already activated 
                                to a format suitable for decoding. Default is True.
        softmax_probs (bool): Whether to use softmax on class scores instead of sigmoid.
                              This is only applicable when `activate_logits = True`. Default is False.

    Returns:
        pred_dicts (List[dict]): A list of length batch_size, containing prediction dictionaries for each image sample in the batch.
                                 The keys of each prediction dictionary are named to be compatible with `torchmetrics.detection.mean_ap`.
                                 They are as follows:
                                    - boxes (torch.Tensor): The predicted bounding boxes in (x_min, y_min, x_max, y_max) format.
                                                            Shape is (num_filtered_bboxes, 4).
                                    - labels (torch.Tensor): The predicted class labels for the bounding boxes in `bboxes`.
                                                             Shape is (num_filtered_bboxes,)
                                    - scores (torch.Tensor): The class probabilities for `labels`. 
                                                             This is defined as P(object) * P(class_i | object) = P(class_i).
                                                             Shape is (num_filtered_bboxes,)

    '''
    device = scale_logits[0].device # Get device to compute on
    
    # --------------------
    # Format Conversions
    # --------------------
    if activate_logits:
        # Applies sigmoid to t_x, t_y (center offsets), t_o (objectness)
        # Either applies sigmoid or softmax to class logits
        scale_preds = postprocess.activate_yolov3_logits(scale_logits = scale_logits, 
                                                         softmax_probs = softmax_probs)
    else:
        # Assumes activations are already applied
        scale_preds = scale_logits

    for preds, anchors, stride in zip(scale_preds, scale_anchors, strides):
        # preds shape: (batch_size, num_anchors, fmap_h, fmap_w, 5 + C)
        anchors = anchors.to(device)
        preds[..., :4] = postprocess.decode_yolov3_bboxes(
            bboxes = preds[..., :4], 
            anchors = anchors, 
            stride = stride,
            return_format = 'xyxy',
            return_units = 'pixel'
        )    

    # Flatten shape: (batch_size, num_anchors * fmap_h * fmap_w, 5 + C)
    flatten_preds = [torch.flatten(preds, start_dim = 1, end_dim = -2) for preds in scale_preds]
    all_preds = torch.concat(flatten_preds, dim = 1) # Shape: (batch_size, num_bboxes, 5 + C)
    
    # --------------------
    # Filtering
    # --------------------
    preds_dicts = []
    # Loop over the batch
    for samp_preds in all_preds:    
        preds_res = {} # Dictionary to store filtered bboxes, scores, and class labels

        # Filter out low predicted object confidence
        samp_preds = samp_preds[samp_preds[:, 4] >= obj_threshold] # Shape: (num_filtered_bboxes, 5 + C)

        bboxes = samp_preds[:, :4] # Shape: (num_filtered_bboxes, 4)
        cls_probs, cls_labels = samp_preds[:, 5:].max(dim = -1) # Both shapes: (num_filtered_bboxes,)

        # Class probability scores
        cls_scores = cls_probs * samp_preds[:, 4]
        
        # Filter duplicate detections with NMS
        keep_idxs = batched_nms(boxes = bboxes, 
                                scores = cls_scores, 
                                idxs = cls_labels, 
                                iou_threshold = nms_threshold)

        preds_res['boxes'] = bboxes[keep_idxs] # Format: XYXY, Units: pixel
        preds_res['labels'] = cls_labels[keep_idxs]
        preds_res['scores'] = cls_scores[keep_idxs]
        preds_dicts.append(preds_res)
        
    return preds_dicts